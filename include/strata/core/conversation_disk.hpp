// The disk tier of --serve's parked conversations (--conversation-disk-dir): the RAM cache's evictions
// (ConversationCache::set_on_evict) are written to session files (conversation_file.hpp, one per layer-split stage)
// by a background thread and read back when a request continues them, within a time limit and a byte budget.
// By default only this process reads its files back, and the folder's files of this tier are removed at start and at
// the end.  `persist`: the files are bound to the session files' identity (model and settings) instead, a small index
// file beside each conversation lists its tokens, and the next start picks them up (no time limit, the budget only).
//
// Цепочки (max_deltas > 0): беседа, которая вернулась на диск, пока там лежит её прошлое состояние (взятое take()
// или вытесненное ранее), пишется не целиком, а дельтой к нему: только блоки полей образа, чьи хеши изменились
// (хвост K/V, свежее running state, новые контрольные точки).  База - прежние сессионные файлы и индекс v1 ровно
// как без цепочек (старый движок видит базу), дельты и их опись - в файле-спутнике .chn.  Цепочка вытесняется и
// уплотняется (переписывается целиком) как одно целое.  Формат - в DELTA.md рядом с исходниками.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace strata::core {

struct ConversationDiskOptions {
    std::string dir;                 ///< empty: the tier is off
    uint64_t budget_bytes = 0;       ///< at most this many bytes of files; 0: off
    uint64_t min_free_bytes = 0;     ///< a write needs the file plus this much free on the disk (session preflight)
    bool persist = false;            ///< keep the files for later runs, bound to `identity`
    SessionFileIdentity identity;    ///< persist: the session files' identity (model and settings fingerprints)
    /// Called with every diagnostic line (no trailing newline), from the caller's or the writer's thread.
    std::function<void(const std::string&)> log;
    /// Tests: called on the writer thread before each write starts.
    std::function<void()> before_write;
    /// --conversation-disk-deltas: сколько дельт держит цепочка, прежде чем следующая запись перепишет образ целиком;
    /// 0 - без цепочек: каждая запись целиком, take() убирает запись с диска (поведение до цепочек).
    size_t max_deltas = 8;
    /// Размер блока, которым дельта сравнивает поля образа (байт); тесты берут маленький.
    uint64_t delta_block_bytes = 256u << 10;
};

/// What the tier did since it started (diagnostics).
struct ConversationDiskStats {
    size_t queued = 0, written = 0, write_failed = 0, ram_hits = 0, disk_hits = 0, read_failed = 0;
    size_t evicted = 0, waits = 0;
    uint64_t bytes_written = 0, bytes_read = 0;
    uint64_t tokens_back = 0;                 ///< prompt tokens given back (not read again) by ram or disk hits
    double write_ms = 0, read_ms = 0, wait_ms = 0;
    size_t deltas = 0;              ///< записей дельтой (входят в written)
    size_t compactions = 0;         ///< цепочек, переписанных целиком по пределу дельт или их объёма
    size_t chain_cuts = 0;          ///< цепочек, откаченных на базу из-за битой или чужой дельты
    uint64_t delta_bytes = 0;       ///< байт дельт (входят в bytes_written)
    uint64_t delta_image_bytes = 0; ///< сколько весили бы те же записи целиком
};

class ConversationDiskTier {
public:
    explicit ConversationDiskTier(ConversationDiskOptions options);
    ~ConversationDiskTier();
    ConversationDiskTier(const ConversationDiskTier&) = delete;
    ConversationDiskTier& operator=(const ConversationDiskTier&) = delete;

    bool enabled() const { return open_; }
    bool persistent() const { return open_ && options_.persist; }
    /// Creates the folder (0700 on POSIX) and starts the writer.  This tier's files a previous run left there are
    /// removed, or with `persist` taken in when their identity matches (the others removed).  False (and the tier
    /// stays off) when the folder cannot be used.
    bool open(std::string& error);

    /// Queues `image` for the writer; its RAM is freed when its write ends, stored or not.  Every checkpoint is written
    /// with it (they hold no K/V), so a prompt that leaves the conversation early still resumes from one.  False when
    /// the tier is off or over budget.
    bool put(SavedConversation&& image);

    struct Match {
        uint64_t id = 0;        ///< the entry (stable while the writer runs; 0 = none)
        int64_t tokens = 0;
        bool live = false;
        uint64_t bytes = 0;     ///< the entry's files (queued: the image's estimate)
        bool queued = false;    ///< still in RAM: take() hands it back without reading a file
    };
    /// The longest prefix of `prompt` an entry holds (its live state or its checkpoint), queued or on the disk, as
    /// ConversationCache::best; ties prefer the newest entry.
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        std::lock_guard<std::mutex> lock(m_);
        Match best;
        auto scan = [&](const Entry& e, bool queued) {
            if (e.cvec != cvec) return;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens || (n == best.tokens && n > 0 && e.id > best.id)) best = {e.id, n, live, e.bytes, queued};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        };
        for (const auto& e : entries_) scan(e, false);
        for (const auto& j : queue_) scan(j.entry, true);
        if (writing_) scan(writing_entry_, false);   // its file appears when the write ends: take() waits for it
        return best;
    }

    /// Hands entry `id` back in `out` (its stage images included) and removes it from the tier either way: from RAM
    /// when it is still queued, else from the disk (waiting for its write when it is being written).  `admit(need,
    /// why)` is asked before each file's payload is allocated (session_file_read's RAM preflight).  `tokens` is what
    /// the request reuses (diagnostics).  On failure `out` is empty and `error` says why.
    /// С цепочками запись, прочитанная с диска, остаётся там (встаёт последней в очередь вытеснения): она - база
    /// дельты, когда беседа вернётся.  Битая дельта откатывает цепочку на базу: take() не удался, но следующий
    /// best() найдёт базу.
    bool take(uint64_t id, int64_t tokens, SavedConversation& out, std::string& error,
              const std::function<bool(uint64_t need_bytes, std::string& why)>& admit = {});

    /// Whether images wait in the queue or are being written (their RAM is not back yet).
    bool pending() const;
    /// Blocks until the queue is empty and no write runs; returns the ms it waited.
    double wait_idle();
    /// #879: every conversation goes - queued, being written (waited for) and on the disk, files too.  Returns how many.
    size_t clear();

    size_t size() const;                    ///< entries on the disk
    uint64_t bytes() const;                 ///< their files' bytes
    ConversationDiskStats stats() const;
    /// One line of totals for the log.
    std::string summary() const;
    /// The file name prefix of this tier (tests).
    static const char* file_prefix() { return "strata-conv-"; }

    /// Подпись поля образа: длина и хеш каждого блока (delta_block_bytes).
    struct FieldSig {
        uint64_t len = 0;
        std::vector<uint64_t> blocks;
    };
    using Table = std::map<uint64_t, FieldSig>;   ///< ключ поля (стадия, вид, номер) -> подпись
    struct Prepared;                              ///< образ, разложенный на поля перед записью (conversation_disk.cpp)

private:
    // Цепочка записи: база (files[0, base_files) - сессионные файлы стадий) и дельты (остальные files, по порядку).
    struct Chain {
        size_t base_files = 0;
        uint64_t base_bytes = 0;
        bool known = false;                      ///< table описывает текущее состояние: можно писать дельту
        uint64_t block = 0;                      ///< размер блока table
        uint64_t digest = 0, base_digest = 0;    ///< отпечаток текущего состояния и базы (сцепка дельт)
        std::shared_ptr<const Table> table;
        ConversationCheckpoint base_live;        ///< ключи базы (есть дельты): на них откатывается цепочка
        std::vector<ConversationCheckpoint> base_checkpoints;
        std::string file;                        ///< persist: опись цепочки (.chn)
    };
    struct Entry {
        uint64_t id = 0;
        std::vector<std::string> files;          ///< the first stage's image, then one per later stage
        ConversationCheckpoint live;             ///< ids and images only: what best() matches
        std::vector<ConversationCheckpoint> checkpoints;
        bool cvec = true;
        uint64_t bytes = 0;                      ///< the files' bytes (queued: the image's estimate)
        std::string index;                       ///< persist: the index file
        int64_t written_s = 0;                   ///< persist: when it was written (seconds since the epoch)
        Chain chain;
        size_t deltas() const { return files.size() - chain.base_files; }
    };
    struct Job {
        Entry entry;
        SavedConversation image;
    };
    void writer();
    void write(Job& job);                                    // without the lock
    bool write_full(Job& job, Prepared* prepared, uint64_t replace, bool compaction, const char* why);
    bool write_delta(Job& job, Prepared& prepared, const Entry& target);
    void remove_locked(size_t index, bool delete_files);     // m_ held
    void drop_file_locked(const std::string& path);         // m_ held: удалить сейчас или после чтения
    size_t find_locked(uint64_t id) const;                   // m_ held: индекс записи или entries_.size()
    bool cut_back_locked(const std::string& bad_delta, std::string& note);   // m_ held
    size_t superseded_locked(const Entry& n, size_t stages) const;            // m_ held
    bool read_chain(const Entry& want, SavedConversation& out, std::string& error,
                    const std::function<bool(uint64_t, std::string&)>& admit, uint64_t& bytes, size_t& bad,
                    bool& memory);
    void log(const std::string& line) const;
    SessionFileIdentity file_identity() const;
    bool write_index(Entry& e, std::string& error) const;
    bool read_index(const std::string& path, Entry& e) const;
    bool write_chain_file(Entry& e, std::string& error) const;
    bool read_chain_file(Entry& e) const;
    std::string scan_folder(size_t& stale, std::error_code& ec);   // open(): take in or remove; its log part

    ConversationDiskOptions options_;
    bool open_ = false;
    uint64_t nonce_ = 0, next_id_ = 0, bytes_ = 0;
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<Entry> entries_;   ///< on the disk, oldest first
    std::deque<Job> queue_;       ///< waiting for the writer, oldest first
    std::multiset<std::string> reading_;   ///< файлы, которые сейчас читает take()
    std::set<std::string> doomed_;         ///< удалить, когда их дочитают
    bool writing_ = false;        ///< the writer holds a job (writing_entry_ is its keys)
    Entry writing_entry_;
    bool stop_ = false;
    std::thread thread_;
    ConversationDiskStats stats_;
};

}  // namespace strata::core
