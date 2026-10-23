// src/core/conversation_disk.cpp - the disk tier of the parked conversations.  See the header.
#include "strata/core/conversation_disk.hpp"

#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <set>
#include <string_view>
#include <system_error>
#include <utility>
#if defined(__linux__)
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strata::core {
namespace {

namespace fs = std::filesystem;

// The second half of every file's identity: with this run's random number as the first, a file another run (or
// another tier) wrote is refused on read.
constexpr uint64_t kTierConfig = 0x5354524154414449ull;   // "STRATADI"

// ids and images only: the part of a checkpoint best() matches against
ConversationCheckpoint key_of(const ConversationCheckpoint& c) {
    ConversationCheckpoint k;
    k.ids = c.ids;
    k.imgs = c.imgs;
    return k;
}

// The payload is the conversation its index names: the same keys and control-vector state, in every stage.
bool holds(const SavedConversation& s, const ConversationCheckpoint& live,
           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
    auto same = [](const ConversationCheckpoint& x, const ConversationCheckpoint& y) {
        return x.ids == y.ids && x.imgs == y.imgs;
    };
    if (s.cvec != cvec || !same(s.live, live) || s.checkpoints.size() != checkpoints.size()) return false;
    for (size_t i = 0; i < checkpoints.size(); ++i)
        if (!same(s.checkpoints[i], checkpoints[i])) return false;
    for (const auto& stage : s.stage_images)
        if (!holds(stage, live, checkpoints, cvec)) return false;
    return true;
}

// a file of this tier (also a temporary one session_file_write left behind: ".<name>.<hex>.tmp")
bool tier_file(const std::string& name) {
    const std::string prefix = ConversationDiskTier::file_prefix();
    return name.rfind(prefix, 0) == 0 || name.rfind("." + prefix, 0) == 0;
}

void remove_file(const std::string& path) {
    std::error_code ec;
    fs::remove(fs::path(path), ec);
}

// The index file of a persistent entry: native-endian fields (the session files refuse a big-endian build too),
// counts before arrays.
constexpr uint64_t kIndexMagic = 0x3158444943525453ull;   // "STRCIDX1"

struct IndexWriter {
    std::string b;
    template<class T> void put(T v) { b.append(reinterpret_cast<const char*>(&v), sizeof v); }
    void str(const std::string& s) { put<uint32_t>((uint32_t) s.size()); b += s; }
    void key(const ConversationCheckpoint& c) {
        put<uint32_t>((uint32_t) c.ids.size());
        b.append(reinterpret_cast<const char*>(c.ids.data()), c.ids.size() * sizeof(int32_t));
        put<uint32_t>((uint32_t) c.imgs.size());
        for (const auto& i : c.imgs) { put<int64_t>(i.start); put<uint64_t>(i.hash); }
    }
};

struct IndexReader {
    std::string_view b;
    size_t at = 0;
    bool ok = true;
    size_t left() const { return b.size() - at; }
    template<class T> T get() {
        T v{};
        if (left() < sizeof v) { ok = false; return v; }
        std::memcpy(&v, b.data() + at, sizeof v);
        at += sizeof v;
        return v;
    }
    std::string str() {
        const uint32_t n = get<uint32_t>();
        if (!ok || left() < n) { ok = false; return {}; }
        at += n;
        return std::string(b.substr(at - n, n));
    }
    bool key(ConversationCheckpoint& c) {
        const uint32_t n = get<uint32_t>();
        if (!ok || left() / sizeof(int32_t) < n) return ok = false;
        c.ids.resize(n);
        std::memcpy(c.ids.data(), b.data() + at, n * sizeof(int32_t));
        at += n * sizeof(int32_t);
        const uint32_t m = get<uint32_t>();
        if (!ok || left() / 16 < m) return ok = false;
        for (uint32_t i = 0; i < m; ++i) c.imgs.push_back({get<int64_t>(), get<uint64_t>()});
        return ok;
    }
};

double mib(uint64_t bytes) { return double(bytes) / (1024.0 * 1024.0); }
double mbps(uint64_t bytes, double ms) { return ms > 0 ? double(bytes) / 1e6 / (ms / 1000.0) : 0.0; }
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

#if defined(__GNUC__)
std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
#endif
std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

// ---- цепочки: поля образа, их подписи, дельта и опись цепочки

constexpr uint64_t kDeltaMagic = 0x31544c4443525453ull;   // "STRCDLT1"
constexpr uint64_t kChainMagic = 0x314e484343525453ull;   // "STRCCHN1"
constexpr uint64_t kDigestSeed = 0x434841494e444947ull;

// Большой массив образа одной стадии: ids, массив running state или часть K/V.  Ключ - стадия, место (живое
// состояние, контрольная точка по числу её токенов, слой K/V по номеру) и вид массива: так поле узнаётся в следующем
// образе той же беседы, даже когда контрольные точки добавились или ушли.
struct Part {
    uint64_t key = 0;
    std::vector<int32_t>* ids = nullptr;
    std::vector<uint8_t>* bytes = nullptr;
    ConversationBuffer* buf = nullptr;
    size_t size() const { return ids ? ids->size() * sizeof(int32_t) : bytes ? bytes->size() : buf->size(); }
    bool resize(size_t n) {
        if (ids) {
            if (n % sizeof(int32_t)) return false;
            ids->resize(n / sizeof(int32_t));
        } else if (bytes) {
            bytes->resize(n);
        } else {
            buf->resize(n);
        }
        return true;
    }
    // fn(указатель, число байт, смещение от начала поля); false прерывает обход
    template<class Fn> bool visit(size_t off, size_t n, Fn fn) {
        if (buf) return buf->visit(off, n, fn);
        if (off > size() || n > size() - off) return false;
        uint8_t* p = ids ? reinterpret_cast<uint8_t*>(ids->data()) : bytes->data();
        return n == 0 || fn(p + off, n, off);
    }
    void take_from(Part& o) {   // поле того же ключа: того же вида
        if (ids) *ids = std::move(*o.ids);
        else if (bytes) *bytes = std::move(*o.bytes);
        else *buf = std::move(*o.buf);
    }
};

constexpr unsigned kSlotLive = 0, kSlotCheckpoint = 1, kSlotKv = 2;
constexpr uint64_t kSubLimit = 1ull << 48;

uint64_t part_key(size_t stage, unsigned slot, unsigned kind, uint64_t sub) {
    return (uint64_t) stage << 56 | (uint64_t) slot << 52 | (uint64_t) kind << 48 | sub;
}

// Поля одной стадии; false - образ не раскладывается на поля (две контрольные точки одной длины, слишком много
// стадий): такой пишется только целиком.
bool stage_parts(SavedConversation& s, size_t stage, std::vector<Part>& out) {
    if (stage > 255) return false;
    auto state = [&](ConversationCheckpoint& c, unsigned slot, uint64_t sub) {
        if (sub >= kSubLimit) return false;
        Part ids;
        ids.key = part_key(stage, slot, 0, sub);
        ids.ids = &c.ids;
        out.push_back(ids);
        std::vector<uint8_t>* arrays[] = {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos};
        for (unsigned k = 0; k < 5; ++k) {
            Part p;
            p.key = part_key(stage, slot, k + 1, sub);
            p.bytes = arrays[k];
            out.push_back(p);
        }
        return true;
    };
    if (!state(s.live, kSlotLive, 0)) return false;
    std::set<size_t> lengths;
    for (auto& c : s.checkpoints)
        if (!lengths.insert(c.ids.size()).second || !state(c, kSlotCheckpoint, c.ids.size())) return false;
    if (s.kv.size() >= kSubLimit) return false;
    for (size_t l = 0; l < s.kv.size(); ++l) {
        auto& kv = s.kv[l];
        ConversationBuffer* bufs[] = {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
        for (unsigned k = 0; k < 5; ++k) {
            Part p;
            p.key = part_key(stage, kSlotKv, k, l);
            p.buf = bufs[k];
            out.push_back(p);
        }
    }
    return true;
}

// Стадии образа по порядку: первая, затем stage_images (вложенных стадий у образа диска нет).
std::vector<SavedConversation*> stages_of(SavedConversation& img) {
    std::vector<SavedConversation*> v{&img};
    for (auto& s : img.stage_images) v.push_back(&s);
    return v;
}

bool image_parts(SavedConversation& img, std::vector<Part>& out) {
    out.clear();
    const auto stages = stages_of(img);
    for (size_t k = 0; k < stages.size(); ++k) {
        if (k > 0 && !stages[k]->stage_images.empty()) return false;
        if (!stage_parts(*stages[k], k, out)) return false;
    }
    return true;
}

// Всё, что сессионный файл хранит кроме больших массивов: геометрия, слои, флаг управляющих векторов, картинки,
// метки LRU, число токенов каждой контрольной точки, заголовки слоёв K/V.
std::string skeleton_of(SavedConversation& img) {
    IndexWriter w;
    const auto stages = stages_of(img);
    w.put<uint32_t>((uint32_t) stages.size());
    for (const SavedConversation* s : stages) {
        for (int64_t g : s->geometry) w.put<int64_t>(g);
        w.put<int64_t>(s->layer_lo);
        w.put<int64_t>(s->layer_hi);
        w.put<uint8_t>(s->cvec ? 1 : 0);
        auto meta = [&](const ConversationCheckpoint& c) {
            w.put<uint64_t>(c.ids.size());
            w.put<uint32_t>((uint32_t) c.imgs.size());
            for (const auto& i : c.imgs) { w.put<int64_t>(i.start); w.put<uint64_t>(i.hash); }
            w.put<uint64_t>(c.used);
        };
        meta(s->live);
        w.put<uint32_t>((uint32_t) s->checkpoints.size());
        for (const auto& c : s->checkpoints) meta(c);
        w.put<uint32_t>((uint32_t) s->kv.size());
        for (const auto& k : s->kv)
            for (int64_t v : {int64_t(k.format), k.cells, k.heads, k.head_dim, k.page_size, k.pooled_rows, k.idx_dim})
                w.put<int64_t>(v);
    }
    return w.b;
}

// Каркас образа из skeleton_of: всё, кроме содержимого больших массивов (ids контрольных точек - нужной длины).
bool skeleton_read(IndexReader& in, std::vector<SavedConversation>& stages) {
    const uint32_t n = in.get<uint32_t>();
    if (!in.ok || n == 0 || n > 256) return false;
    stages.assign(n, SavedConversation{});
    for (auto& s : stages) {
        for (auto& g : s.geometry) g = in.get<int64_t>();
        s.layer_lo = in.get<int64_t>();
        s.layer_hi = in.get<int64_t>();
        s.cvec = in.get<uint8_t>() != 0;
        auto meta = [&](ConversationCheckpoint& c) {
            const uint64_t tokens = in.get<uint64_t>();
            const uint32_t imgs = in.get<uint32_t>();
            if (!in.ok || tokens > (1ull << 32) || in.left() / 16 < imgs) return in.ok = false;
            c.ids.resize((size_t) tokens);
            for (uint32_t i = 0; i < imgs; ++i) c.imgs.push_back({in.get<int64_t>(), in.get<uint64_t>()});
            c.used = in.get<uint64_t>();
            return in.ok;
        };
        if (!meta(s.live)) return false;
        const uint32_t nck = in.get<uint32_t>();
        if (!in.ok || in.left() / 20 < nck) return false;
        s.checkpoints.resize(nck);
        for (auto& c : s.checkpoints) if (!meta(c)) return false;
        const uint32_t nkv = in.get<uint32_t>();
        if (!in.ok || in.left() / 56 < nkv) return false;
        s.kv.resize(nkv);
        for (auto& k : s.kv) {
            const int64_t format = in.get<int64_t>();
            if (format < INT32_MIN || format > INT32_MAX) return false;
            k.format = (int) format;
            k.cells = in.get<int64_t>(); k.heads = in.get<int64_t>(); k.head_dim = in.get<int64_t>();
            k.page_size = in.get<int64_t>(); k.pooled_rows = in.get<int64_t>(); k.idx_dim = in.get<int64_t>();
        }
    }
    return in.ok;
}

uint64_t block_count(uint64_t len, uint64_t block) { return (len + block - 1) / block; }

uint64_t part_hash(Part& p, size_t off, size_t n) {
    SessionHasher h;
    p.visit(off, n, [&](const uint8_t* q, size_t c, size_t) { h.update(q, c); return true; });
    return h.digest();
}

ConversationDiskTier::FieldSig sign(Part& p, uint64_t block) {
    ConversationDiskTier::FieldSig f;
    f.len = p.size();
    const uint64_t n = block_count(f.len, block);
    f.blocks.reserve((size_t) n);
    for (uint64_t i = 0; i < n; ++i) {
        const uint64_t off = i * block;
        f.blocks.push_back(part_hash(p, (size_t) off, (size_t) std::min(block, f.len - off)));
    }
    return f;
}

// Отпечаток состояния: каркас и подписи всех полей.  Им сцеплены база и дельты цепочки.
uint64_t state_digest(const std::string& skeleton, const ConversationDiskTier::Table& table) {
    SessionHasher h(kDigestSeed);
    h.update(skeleton.data(), skeleton.size());
    for (const auto& [key, f] : table) {
        h.update(&key, sizeof key);
        h.update(&f.len, sizeof f.len);
        h.update(f.blocks.data(), f.blocks.size() * sizeof(uint64_t));
    }
    return h.digest();
}

// `a` - начало `b`: те же первые токены и картинки
bool key_prefix(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return !a.ids.empty() && a.ids.size() <= b.ids.size() && a.imgs.size() <= b.imgs.size() &&
           std::equal(a.ids.begin(), a.ids.end(), b.ids.begin()) &&
           std::equal(a.imgs.begin(), a.imgs.end(), b.imgs.begin());
}

bool same_key(const ConversationCheckpoint& x, const ConversationCheckpoint& y) {
    return x.ids == y.ids && x.imgs == y.imgs;
}

// Файл дельты или описи: пишется во временный рядом, с хешем всего содержимого в последних 8 байтах.
struct FileOut {
    std::FILE* f = nullptr;
    SessionHasher hash;
    uint64_t n = 0;
    bool ok = true;
    std::vector<char> vbuf;
    bool open(const fs::path& temp) {
        f = std::fopen(temp.string().c_str(), "wb");
        if (!f) return ok = false;
        vbuf.resize(1u << 20);
        std::setvbuf(f, vbuf.data(), _IOFBF, vbuf.size());
        return true;
    }
    void raw(const void* p, size_t c) {
        if (!ok || !c) return;
        hash.update(p, c);
        ok = std::fwrite(p, 1, c, f) == c;
        n += c;
    }
    template<class T> void put(T v) { raw(&v, sizeof v); }
    bool close_with_hash() {
        if (ok) {
            const uint64_t h = hash.digest();
            ok = std::fwrite(&h, 1, sizeof h, f) == sizeof h;
            n += sizeof h;
        }
        if (f && std::fclose(f) != 0) ok = false;
        f = nullptr;
        return ok;
    }
    ~FileOut() { if (f) std::fclose(f); }
};

bool publish(const fs::path& temp, const fs::path& path, std::string& error) {
    std::error_code ec;
    fs::rename(temp, path, ec);   // читатель видит файл целиком или никакой
    if (ec) {
        error = "cannot publish " + path.filename().string() + ": " + ec.message();
        remove_file(temp.string());
        return false;
    }
    return true;
}

// Файл целиком в память; хеш в последних 8 байтах должен сойтись с остальным содержимым (он отрезается).
bool read_hashed(const std::string& path, std::string& b, std::string& error) {
    const std::string name = fs::path(path).filename().string();
    std::error_code ec;
    const uint64_t size = fs::file_size(fs::path(path), ec);
    std::ifstream f(fs::path(path), std::ios::binary);
    if (ec || !f) { error = "cannot open " + name; return false; }
    b.resize((size_t) size);
    if (!f.read(b.data(), (std::streamsize) size)) { error = "cannot read " + name; return false; }
    if (b.size() < 16) { error = name + " is cut short"; return false; }
    uint64_t want = 0;
    std::memcpy(&want, b.data() + b.size() - 8, 8);
    b.resize(b.size() - 8);
    if (session_hash64(b.data(), b.size(), 0) != want) {
        error = name + " is damaged (its hash does not match)";
        return false;
    }
    return true;
}

}  // namespace

// Образ, разложенный на поля перед записью: подписи полей (их сравнит следующая дельта) и каркас.
struct ConversationDiskTier::Prepared {
    std::vector<Part> parts;               // указывают в job.image
    std::string skeleton;
    std::shared_ptr<Table> table;
    uint64_t digest = 0, block = 0;
    uint64_t image_bytes = 0;              // поля и каркас: примерно вес образа целиком
    std::vector<std::vector<uint32_t>> changed;   // для дельты: изменившиеся блоки каждого поля
    uint64_t delta_bytes = 0;                     // и размер её файла
};

namespace {

bool prepare(SavedConversation& img, uint64_t block, ConversationDiskTier::Prepared& p) {
    if (block == 0 || !image_parts(img, p.parts)) return false;
    p.block = block;
    p.table = std::make_shared<ConversationDiskTier::Table>();
    p.skeleton = skeleton_of(img);
    p.image_bytes = p.skeleton.size();
    for (auto& part : p.parts) {
        if (block_count(part.size(), block) > UINT32_MAX) return false;
        p.image_bytes += part.size();
        (*p.table)[part.key] = sign(part, block);
    }
    p.digest = state_digest(p.skeleton, *p.table);
    return true;
}

// Какие блоки дельта пишет относительно `old`, и размер её файла.
void plan_delta(ConversationDiskTier::Prepared& p, const ConversationDiskTier::Table& old) {
    p.changed.assign(p.parts.size(), {});
    uint64_t bytes = 7 * 8 + p.skeleton.size() + 4 + 8;   // заголовок, каркас, число полей, хеш
    for (size_t i = 0; i < p.parts.size(); ++i) {
        const auto& now = p.table->at(p.parts[i].key);
        const auto was = old.find(p.parts[i].key);
        bytes += 8 + 8 + 4;
        for (size_t b = 0; b < now.blocks.size(); ++b) {
            const uint64_t len = std::min<uint64_t>(p.block, now.len - b * p.block);
            const bool same = was != old.end() && b < was->second.blocks.size() &&
                              std::min<uint64_t>(p.block, was->second.len - b * p.block) == len &&
                              was->second.blocks[b] == now.blocks[b];
            if (same) continue;
            p.changed[i].push_back((uint32_t) b);
            bytes += 4 + len;
        }
    }
    p.delta_bytes = bytes;
}

}  // namespace

ConversationDiskTier::ConversationDiskTier(ConversationDiskOptions options) : options_(std::move(options)) {
    std::random_device rd;
    nonce_ = ((uint64_t) rd() << 32) ^ rd() ^ (uint64_t) std::chrono::system_clock::now().time_since_epoch().count();
}

ConversationDiskTier::~ConversationDiskTier() {
    if (open_ && options_.persist) wait_idle();   // what is queued belongs to the next start
    size_t dropped = 0;
    {
        std::lock_guard<std::mutex> lock(m_);
        stop_ = true;
        dropped = queue_.size();
        queue_.clear();   // not written: dropped, as the RAM cache would have
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    std::lock_guard<std::mutex> lock(m_);
    if (!open_) return;
    if (options_.persist) {
        log(fmt("end: %zu conversations kept on the disk for the next start (%.0f MiB)", entries_.size(), mib(bytes_)));
        return;
    }
    if (dropped || !entries_.empty())
        log(fmt("end: %zu queued dropped, %zu conversations removed from the disk", dropped, entries_.size()));
    while (!entries_.empty()) remove_locked(0, true);
}

SessionFileIdentity ConversationDiskTier::file_identity() const {
    return options_.persist ? options_.identity : SessionFileIdentity{nonce_, kTierConfig};
}

bool ConversationDiskTier::write_index(Entry& e, std::string& error) const {
    char name[96];
    std::snprintf(name, sizeof name, "%s%016llx-%llu.idx", file_prefix(), (unsigned long long) nonce_,
                  (unsigned long long) e.id);
    const fs::path path = e.index.empty() ? fs::path(options_.dir) / name : fs::path(e.index);
    const fs::path temp = path.parent_path() / ("." + path.filename().string() + ".tmp");
    e.written_s = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    // индекс описывает базу (формат v1 как до цепочек): движок без цепочек поднимет её, дельты описывает .chn
    const bool chained = e.deltas() > 0;
    const ConversationCheckpoint& live = chained ? e.chain.base_live : e.live;
    const std::vector<ConversationCheckpoint>& checkpoints = chained ? e.chain.base_checkpoints : e.checkpoints;
    const SessionFileIdentity id = file_identity();
    IndexWriter out;
    out.put(kIndexMagic);
    out.put(id.model);
    out.put(id.config);
    out.put<int64_t>(e.written_s);
    out.put<uint8_t>(e.cvec ? 1 : 0);
    out.put<uint64_t>(chained ? e.chain.base_bytes : e.bytes);
    out.put<uint32_t>((uint32_t) e.chain.base_files);
    for (size_t k = 0; k < e.chain.base_files; ++k) out.str(fs::path(e.files[k]).filename().string());
    out.key(live);
    out.put<uint32_t>((uint32_t) checkpoints.size());
    for (const auto& c : checkpoints) out.key(c);
    {
        std::ofstream f(temp, std::ios::binary | std::ios::trunc);
        f.write(out.b.data(), (std::streamsize) out.b.size());
        if (!f) {
            error = "cannot write the index " + temp.string();
            remove_file(temp.string());
            return false;
        }
    }
    std::error_code ec;
    fs::rename(temp, path, ec);   // a reader sees the whole index or none
    if (ec) {
        error = "cannot publish the index: " + ec.message();
        remove_file(temp.string());
        return false;
    }
    e.index = path.string();
    return true;
}

bool ConversationDiskTier::read_index(const std::string& path, Entry& e) const {
    std::ifstream f(fs::path(path), std::ios::binary);
    const std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    IndexReader in{b};
    const SessionFileIdentity id = file_identity();
    if (in.get<uint64_t>() != kIndexMagic || in.get<uint64_t>() != id.model || in.get<uint64_t>() != id.config)
        return false;
    e.written_s = in.get<int64_t>();
    e.cvec = in.get<uint8_t>() != 0;
    e.bytes = in.get<uint64_t>();
    const uint32_t nfiles = in.get<uint32_t>();
    for (uint32_t i = 0; in.ok && i < nfiles; ++i) e.files.push_back((fs::path(options_.dir) / in.str()).string());
    if (!in.key(e.live)) return false;
    const uint32_t nck = in.get<uint32_t>();
    for (uint32_t i = 0; in.ok && i < nck; ++i) {
        ConversationCheckpoint c;
        if (in.key(c)) e.checkpoints.push_back(std::move(c));
    }
    if (!in.ok || in.left() != 0 || e.files.empty()) return false;
    std::error_code ec;
    for (const auto& file : e.files)
        if (!fs::is_regular_file(fs::path(file), ec)) return false;
    e.index = path;
    e.chain.base_files = e.files.size();
    e.chain.base_bytes = e.bytes;
    return true;
}

// Опись цепочки рядом с индексом (то же имя, .chn): файлы базы и дельт, ключи и подписи текущего состояния.
bool ConversationDiskTier::write_chain_file(Entry& e, std::string& error) const {
    if (!options_.persist || e.index.empty()) return true;
    const fs::path path = fs::path(e.index).replace_extension(".chn");
    const fs::path temp = path.parent_path() / ("." + path.filename().string() + ".tmp");
    e.written_s = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const SessionFileIdentity id = file_identity();
    IndexWriter out;
    out.put(kChainMagic);
    out.put(id.model);
    out.put(id.config);
    out.put<int64_t>(e.written_s);
    out.put<uint64_t>(e.bytes);
    out.put<uint32_t>((uint32_t) e.chain.base_files);
    out.put<uint32_t>((uint32_t) e.files.size());
    for (const auto& f : e.files) out.str(fs::path(f).filename().string());
    out.put<uint8_t>(e.chain.known ? 1 : 0);
    out.put<uint64_t>(e.chain.block);
    out.put<uint64_t>(e.chain.digest);
    out.put<uint64_t>(e.chain.base_digest);
    out.key(e.live);
    out.put<uint32_t>((uint32_t) e.checkpoints.size());
    for (const auto& c : e.checkpoints) out.key(c);
    const ConversationDiskTier::Table empty;
    const auto& table = e.chain.known && e.chain.table ? *e.chain.table : empty;
    out.put<uint32_t>((uint32_t) table.size());
    for (const auto& [key, f] : table) {
        out.put<uint64_t>(key);
        out.put<uint64_t>(f.len);
        out.put<uint32_t>((uint32_t) f.blocks.size());
        out.b.append(reinterpret_cast<const char*>(f.blocks.data()), f.blocks.size() * sizeof(uint64_t));
    }
    FileOut file;
    if (!file.open(temp)) {
        error = "cannot create " + temp.string();
        return false;
    }
    file.raw(out.b.data(), out.b.size());
    if (!file.close_with_hash()) {
        error = "cannot write " + temp.string();
        remove_file(temp.string());
        return false;
    }
    if (!publish(temp, path, error)) return false;
    e.chain.file = path.string();
    return true;
}

// open(): опись цепочки записи, чей индекс уже прочитан.  Нет описи - база без дельт; опись не годится (другая
// база, нет дельты, повреждена) - тоже база без дельт, а опись и дельты уйдут как лишние файлы.
bool ConversationDiskTier::read_chain_file(Entry& e) const {
    const fs::path path = fs::path(e.index).replace_extension(".chn");
    std::error_code ec;
    if (!fs::is_regular_file(path, ec)) return true;
    std::string b, error;
    if (!read_hashed(path.string(), b, error)) return false;
    IndexReader in{b};
    const SessionFileIdentity id = file_identity();
    if (in.get<uint64_t>() != kChainMagic || in.get<uint64_t>() != id.model || in.get<uint64_t>() != id.config)
        return false;
    const int64_t written_s = in.get<int64_t>();
    const uint64_t bytes = in.get<uint64_t>();
    const uint32_t base_files = in.get<uint32_t>();
    const uint32_t nfiles = in.get<uint32_t>();
    if (!in.ok || base_files != e.files.size() || nfiles < base_files) return false;
    std::vector<std::string> files;
    for (uint32_t i = 0; in.ok && i < nfiles; ++i) files.push_back((fs::path(options_.dir) / in.str()).string());
    if (!in.ok) return false;
    for (uint32_t i = 0; i < base_files; ++i)
        if (fs::path(files[i]).filename() != fs::path(e.files[i]).filename()) return false;   // опись другой базы
    for (uint32_t i = base_files; i < nfiles; ++i)
        if (!fs::is_regular_file(fs::path(files[i]), ec)) return false;
    Chain chain;
    chain.base_files = base_files;
    chain.base_bytes = e.bytes;
    chain.known = in.get<uint8_t>() != 0;
    chain.block = in.get<uint64_t>();
    chain.digest = in.get<uint64_t>();
    chain.base_digest = in.get<uint64_t>();
    ConversationCheckpoint live;
    std::vector<ConversationCheckpoint> checkpoints;
    if (!in.key(live)) return false;
    const uint32_t nck = in.get<uint32_t>();
    for (uint32_t i = 0; in.ok && i < nck; ++i) {
        ConversationCheckpoint c;
        if (in.key(c)) checkpoints.push_back(std::move(c));
    }
    auto table = std::make_shared<Table>();
    const uint32_t nfields = in.get<uint32_t>();
    for (uint32_t i = 0; in.ok && i < nfields; ++i) {
        const uint64_t key = in.get<uint64_t>();
        FieldSig f;
        f.len = in.get<uint64_t>();
        const uint32_t nb = in.get<uint32_t>();
        if (!in.ok || in.left() / 8 < nb) return false;
        f.blocks.resize(nb);
        std::memcpy(f.blocks.data(), in.b.data() + in.at, nb * sizeof(uint64_t));
        in.at += nb * sizeof(uint64_t);
        (*table)[key] = std::move(f);
    }
    if (!in.ok || in.left() != 0 || (chain.known && (chain.block == 0 || table->empty()))) return false;
    if (nfiles > base_files && !chain.known) return false;   // дельты без подписей не собрать
    if (chain.known) chain.table = std::move(table);
    chain.file = path.string();
    if (nfiles > base_files) {   // ключи индекса - ключи базы
        chain.base_live = std::move(e.live);
        chain.base_checkpoints = std::move(e.checkpoints);
    }
    e.live = std::move(live);
    e.checkpoints = std::move(checkpoints);
    e.files = std::move(files);
    e.bytes = bytes;
    e.written_s = written_s;
    e.chain = std::move(chain);
    return true;
}

std::string ConversationDiskTier::scan_folder(size_t& stale, std::error_code& ec) {
    std::vector<Entry> found;
    std::vector<std::string> others;
    size_t refused = 0, chains = 0;
    for (fs::directory_iterator it(fs::path(options_.dir), ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec) || !tier_file(it->path().filename().string())) continue;
        Entry e;
        const bool index = it->path().extension() == ".idx";
        const bool taken = index && read_index(it->path().string(), e);
        if (taken) {
            if (options_.max_deltas > 0 && !read_chain_file(e)) {   // база без дельт
                e.chain.file.clear();
                e.chain.known = false;
            }
            chains += e.deltas() > 0;
            found.push_back(std::move(e));
        } else {
            others.push_back(it->path().string());
        }
        refused += index && !taken;
    }
    // the files of the entries taken in stay; the rest goes (refused indexes and their files, interrupted writes)
    std::set<std::string> kept;
    for (const auto& e : found) {
        kept.insert(e.files.begin(), e.files.end());
        if (!e.chain.file.empty()) kept.insert(e.chain.file);
    }
    for (const auto& p : others)
        if (!kept.count(p)) { remove_file(p); ++stale; }
    std::sort(found.begin(), found.end(), [](const Entry& a, const Entry& b) { return a.written_s < b.written_s; });
    for (auto& e : found) {
        e.id = ++next_id_;
        bytes_ += e.bytes;
        entries_.push_back(std::move(e));
    }
    size_t over = 0;
    while (!entries_.empty() && bytes_ > options_.budget_bytes) {
        remove_locked(0, true);
        ++over;
    }
    return fmt("persistent (budget only): %zu conversations from earlier runs (%.0f MiB, %zu with deltas), %zu refused "
               "(another model, engine version or settings), %zu over the budget removed", entries_.size(), mib(bytes_),
               chains, refused, over);
}

void ConversationDiskTier::log(const std::string& line) const {
    if (options_.log) options_.log(line);
    else std::fprintf(stderr, "strata serve: conversation disk: %s\n", line.c_str());
}

bool ConversationDiskTier::open(std::string& error) {
    open_ = false;
    if (options_.dir.empty() || options_.budget_bytes == 0) return true;   // off: nothing to open
    std::error_code ec;
    const fs::path dir(options_.dir);
    fs::create_directories(dir, ec);
    if (ec || !fs::is_directory(dir, ec)) {
        error = "cannot create the folder " + options_.dir + (ec ? ": " + ec.message() : std::string());
        return false;
    }
#ifndef _WIN32
    fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace, ec);   // the files hold conversations
    ec.clear();
#endif
    size_t stale = 0;
    const std::string taken_in = scan_folder(stale, ec);
    if (ec) {
        error = "cannot list the folder " + options_.dir + ": " + ec.message();
        return false;
    }
    open_ = true;
    thread_ = std::thread([this] { writer(); });
    const std::string kept = options_.persist ? taken_in : "removed at the end";
    const std::string chains = options_.max_deltas
        ? fmt("deltas: up to %zu per chain, %.0f KiB blocks", options_.max_deltas, double(options_.delta_block_bytes) / 1024)
        : std::string("deltas off");
    log(fmt("on: %s, budget %.0f MiB, %s; %s; %zu stale files removed", options_.dir.c_str(),
            mib(options_.budget_bytes), kept.c_str(), chains.c_str(), stale));
    return true;
}

void ConversationDiskTier::drop_file_locked(const std::string& path) {
    if (reading_.count(path)) doomed_.insert(path);   // take() ещё читает: удалит, когда дочитает
    else remove_file(path);
}

size_t ConversationDiskTier::find_locked(uint64_t id) const {
    for (size_t i = 0; i < entries_.size(); ++i)
        if (id != 0 && entries_[i].id == id) return i;
    return entries_.size();
}

void ConversationDiskTier::remove_locked(size_t index, bool delete_files) {
    Entry& e = entries_.at(index);
    if (!e.index.empty()) remove_file(e.index);   // first, always: an entry gone from the list is not taken in again
    if (!e.chain.file.empty()) remove_file(e.chain.file);
    if (delete_files)
        for (const auto& f : e.files) drop_file_locked(f);
    bytes_ -= e.bytes;
    entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
}

// Запись, которую образ `n` заменяет: та же беседа ход-другой назад - её живое состояние начинает живое состояние
// `n`, или её самую глубокую контрольную точку `n` держит (клиент переписал хвост; как drop_superseded кэша в RAM).
// Из нескольких - самая длинная, при равенстве - новее.
size_t ConversationDiskTier::superseded_locked(const Entry& n, size_t stages) const {
    size_t best = entries_.size();
    for (size_t i = 0; i < entries_.size(); ++i) {
        const Entry& e = entries_[i];
        if (e.cvec != n.cvec || e.chain.base_files != stages) continue;
        bool sup = key_prefix(e.live, n.live);
        if (!sup) {
            const ConversationCheckpoint* deepest = nullptr;
            for (const auto& c : e.checkpoints)
                if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
            if (deepest && !deepest->ids.empty()) {
                sup = same_key(*deepest, n.live);
                for (const auto& c : n.checkpoints) sup = sup || same_key(*deepest, c);
            }
        }
        if (sup && (best == entries_.size() || e.live.ids.size() >= entries_[best].live.ids.size())) best = i;
    }
    return best;
}

bool ConversationDiskTier::put(SavedConversation&& image) {
    if (!open_) return false;
    Job job;
    job.image = std::move(image);
    job.entry.live = key_of(job.image.live);
    for (const auto& c : job.image.checkpoints) job.entry.checkpoints.push_back(key_of(c));
    job.entry.cvec = job.image.cvec;
    job.entry.bytes = job.image.bytes();
    const size_t tokens = job.entry.live.ids.size();
    if (job.entry.bytes > options_.budget_bytes) {
        log(fmt("dropped %zu tokens: %.0f MiB is over the disk budget", tokens, mib(job.entry.bytes)));
        std::lock_guard<std::mutex> lock(m_);
        ++stats_.write_failed;
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(m_);
        job.entry.id = ++next_id_;
        ++stats_.queued;
    }
    size_t depth = 0;
    uint64_t queued_bytes = 0;
    const uint64_t estimate = job.entry.bytes;
    {
        std::lock_guard<std::mutex> lock(m_);
        queue_.push_back(std::move(job));
        depth = queue_.size();
        for (const auto& j : queue_) queued_bytes += j.entry.bytes;
    }
    cv_.notify_all();
    log(fmt("queued %zu tokens (%.0f MiB in RAM until written; queue %zu, %.0f MiB)", tokens, mib(estimate), depth,
            mib(queued_bytes)));
    return true;
}

void ConversationDiskTier::writer() {
#if defined(__linux__)
    // below the serving threads: the write is background work, decode and prompt reading come first
    setpriority(PRIO_PROCESS, (id_t) syscall(SYS_gettid), 10);
#endif
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(m_);
            cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            job = std::move(queue_.front());
            queue_.pop_front();
            writing_ = true;
            writing_entry_ = job.entry;   // best() still sees it; take() waits for its file
        }
        write(job);
        {
            std::lock_guard<std::mutex> lock(m_);
            writing_ = false;
            writing_entry_ = Entry{};
        }
        cv_.notify_all();
    }
}

void ConversationDiskTier::write(Job& job) {
    if (options_.before_write) options_.before_write();
    if (options_.max_deltas == 0) {   // без цепочек: как до них
        write_full(job, nullptr, 0, false, nullptr);
        return;
    }
    // подписи полей: по ним эту запись сравнит следующая дельта, а эту - с её прошлым состоянием
    Prepared p;
    const bool keyed = prepare(job.image, options_.delta_block_bytes, p);
    Prepared* prepared = keyed ? &p : nullptr;
    Entry target;
    bool have = false;
    {
        std::lock_guard<std::mutex> lock(m_);
        const size_t i = superseded_locked(job.entry, 1 + job.image.stage_images.size());
        if (i < entries_.size()) {
            target = entries_[i];
            have = true;
        }
    }
    if (!have) {
        write_full(job, prepared, 0, false, nullptr);
        return;
    }
    std::string why;
    if (!keyed) {
        why = "the image does not split into fields";
    } else if (!target.chain.known || target.chain.block != p.block || !target.chain.table) {
        why = "the copy on the disk has no field table";
    } else if (target.deltas() >= options_.max_deltas) {
        why = fmt("the chain holds %zu deltas", target.deltas());
    } else {
        plan_delta(p, *target.chain.table);
        const uint64_t deltas = target.bytes - target.chain.base_bytes;
        if (deltas + p.delta_bytes > target.chain.base_bytes)
            why = fmt("its deltas would outweigh the base: %.0f + %.0f MiB > %.0f MiB", mib(deltas), mib(p.delta_bytes),
                      mib(target.chain.base_bytes));
    }
    if (!why.empty()) {
        write_full(job, prepared, target.id, keyed && target.chain.known, why.c_str());
        return;
    }
    if (!write_delta(job, p, target)) write_full(job, prepared, 0, false, nullptr);
}

bool ConversationDiskTier::write_full(Job& job, Prepared* prepared, uint64_t replace, bool compaction,
                                      const char* why) {
    const size_t tokens = job.entry.live.ids.size();
    size_t made_room = 0;
    {
        std::lock_guard<std::mutex> lock(m_);
        // заменяемая запись уйдёт после записи: место под образ берётся у остальных
        const size_t r = find_locked(replace);
        const uint64_t held = r < entries_.size() ? entries_[r].bytes : 0;
        while (bytes_ - held + job.entry.bytes > options_.budget_bytes) {
            size_t victim = 0;
            while (victim < entries_.size() && entries_[victim].id == replace) ++victim;
            if (victim == entries_.size()) break;
            remove_locked(victim, true);
            ++made_room;
        }
        stats_.evicted += made_room;
    }
    const auto t0 = std::chrono::steady_clock::now();
    // one session file per image: the first stage's, then each later stage's (the writer takes no nested images)
    std::vector<SavedConversation> parts = std::move(job.image.stage_images);
    job.image.stage_images.clear();
    const SessionFileIdentity id = file_identity();
    SessionWriteOptions wo;
    wo.min_free_bytes = options_.min_free_bytes;
    wo.durable = false;   // a cache: a file a power loss damages fails its hash check and is read again from the prompt
    Entry e = std::move(job.entry);
    e.bytes = 0;
    std::string error;
    auto write_one = [&](const SavedConversation& img, size_t k) -> bool {
        char name[96];
        std::snprintf(name, sizeof name, "%s%016llx-%llu-%zu.bin", file_prefix(), (unsigned long long) nonce_,
                      (unsigned long long) e.id, k);
        const std::string path = (fs::path(options_.dir) / name).string();
        size_t written = 0;
        if (!session_file_write(path, img, id, written, error, wo)) {
            remove_file(path);
            return false;
        }
        e.files.push_back(path);
        e.bytes += written;
        return true;
    };
    bool ok = write_one(job.image, 0);
    for (size_t k = 0; ok && k < parts.size(); ++k) ok = write_one(parts[k], k + 1);
    e.chain = Chain{};
    e.chain.base_files = e.files.size();
    e.chain.base_bytes = e.bytes;
    if (prepared) {
        e.chain.known = true;
        e.chain.block = prepared->block;
        e.chain.digest = e.chain.base_digest = prepared->digest;
        e.chain.table = prepared->table;
    }
    if (ok && options_.persist) ok = write_index(e, error);   // after the images: an index names complete files only
    if (ok && options_.persist && options_.max_deltas) ok = write_chain_file(e, error);
    job.image = SavedConversation{};   // the RAM goes back now, written or not
    parts.clear();
    if (prepared) prepared->parts.clear();
    const double ms = ms_since(t0);
    if (!ok) {
        if (!e.index.empty()) remove_file(e.index);
        for (const auto& f : e.files) remove_file(f);
        {
            std::lock_guard<std::mutex> lock(m_);
            ++stats_.write_failed;
        }
        log(fmt("dropped %zu tokens: the write failed after %.0f ms (%s)", tokens, ms, error.c_str()));
        return true;
    }
    const uint64_t b = e.bytes;
    const size_t nfiles = e.files.size();
    size_t on_disk = 0, replaced_deltas = 0;
    uint64_t disk_bytes = 0;
    int64_t replaced_tokens = -1;
    {
        std::lock_guard<std::mutex> lock(m_);
        if (const size_t r = find_locked(replace); r < entries_.size()) {   // прошлое состояние этой беседы уходит
            replaced_tokens = (int64_t) entries_[r].live.ids.size();
            replaced_deltas = entries_[r].deltas();
            remove_locked(r, true);
            if (compaction) ++stats_.compactions;
        }
        bytes_ += e.bytes;
        entries_.push_back(std::move(e));
        ++stats_.written;
        stats_.bytes_written += b;
        stats_.write_ms += ms;
        on_disk = entries_.size();
        disk_bytes = bytes_;
    }
    std::string note;
    if (replaced_tokens >= 0)
        note = compaction ? fmt("; compaction: replaces its chain of %lld tokens and %zu deltas (%s)",
                                (long long) replaced_tokens, replaced_deltas, why ? why : "")
                          : fmt("; replaces its copy of %lld tokens (%s)", (long long) replaced_tokens, why ? why : "");
    log(fmt("wrote %zu tokens: %.0f MiB in %zu file%s, %.0f ms (%.0f MB/s); on disk %zu, %.0f MiB; %zu removed for "
            "the budget%s", tokens, mib(b), nfiles, nfiles == 1 ? "" : "s", ms, mbps(b, ms), on_disk,
            mib(disk_bytes), made_room, note.c_str()));
    return true;
}

// Дельта к цепочке `target`: каркас, полный список полей с длинами и только изменившиеся блоки.  false - запись
// `target` ушла, пока писалась дельта (образ ещё цел: вызывающий пишет его целиком).
bool ConversationDiskTier::write_delta(Job& job, Prepared& p, const Entry& target) {
    const size_t tokens = job.entry.live.ids.size();
    size_t made_room = 0;
    {
        std::lock_guard<std::mutex> lock(m_);
        while (bytes_ + p.delta_bytes > options_.budget_bytes) {
            size_t victim = 0;
            while (victim < entries_.size() && entries_[victim].id == target.id) ++victim;
            if (victim == entries_.size()) break;
            remove_locked(victim, true);
            ++made_room;
        }
        stats_.evicted += made_room;
    }
    const auto t0 = std::chrono::steady_clock::now();
    char name[96];
    std::snprintf(name, sizeof name, "%s%016llx-%llu-delta.bin", file_prefix(), (unsigned long long) nonce_,
                  (unsigned long long) job.entry.id);
    const fs::path path = fs::path(options_.dir) / name;
    const fs::path temp = fs::path(options_.dir) / ("." + std::string(name) + ".tmp");
    std::string error;
    bool ok = true;
    {
        std::error_code ec;
        const auto space = fs::space(fs::path(options_.dir), ec);
        if (!ec && space.available < p.delta_bytes + options_.min_free_bytes) {
            ok = false;
            error = fmt("no room on the disk: %.0f MiB free, %.0f MiB needed", mib(space.available),
                        mib(p.delta_bytes + options_.min_free_bytes));
        }
    }
    FileOut f;
    if (ok && !f.open(temp)) {
        ok = false;
        error = "cannot create " + temp.string() + ": " + std::strerror(errno);
    }
    if (ok) {
        const SessionFileIdentity id = file_identity();
        f.put(kDeltaMagic);
        f.put(id.model);
        f.put(id.config);
        f.put<uint64_t>(p.block);
        f.put<uint64_t>(target.chain.digest);
        f.put<uint64_t>(p.digest);
        f.put<uint64_t>(p.skeleton.size());
        f.raw(p.skeleton.data(), p.skeleton.size());
        f.put<uint32_t>((uint32_t) p.parts.size());
        for (size_t i = 0; i < p.parts.size() && f.ok; ++i) {
            Part& part = p.parts[i];
            const uint64_t len = part.size();
            f.put<uint64_t>(part.key);
            f.put<uint64_t>(len);
            f.put<uint32_t>((uint32_t) p.changed[i].size());
            for (const uint32_t b : p.changed[i]) {
                const uint64_t off = (uint64_t) b * p.block;
                f.put<uint32_t>(b);
                part.visit((size_t) off, (size_t) std::min<uint64_t>(p.block, len - off),
                           [&](const uint8_t* q, size_t c, size_t) { f.raw(q, c); return f.ok; });
            }
        }
        ok = f.close_with_hash();
        if (!ok) error = "cannot write " + temp.string() + ": " + std::strerror(errno);
        if (ok && f.n != p.delta_bytes) {
            ok = false;
            error = fmt("the delta came to %llu bytes, %llu planned", (unsigned long long) f.n,
                        (unsigned long long) p.delta_bytes);
        }
        if (ok) ok = publish(temp, path, error);
        else remove_file(temp.string());
    }
    const double ms = ms_since(t0);
    if (!ok) {
        job.image = SavedConversation{};
        p.parts.clear();
        {
            std::lock_guard<std::mutex> lock(m_);
            ++stats_.write_failed;
        }
        log(fmt("dropped %zu tokens: the delta write failed after %.0f ms (%s)", tokens, ms, error.c_str()));
        return true;
    }
    size_t on_disk = 0, links = 0;
    uint64_t disk_bytes = 0;
    int64_t was = 0;
    {
        std::lock_guard<std::mutex> lock(m_);
        const size_t i = find_locked(target.id);
        if (i == entries_.size()) {   // откат цепочки или промах take() убрали её, пока писалась дельта
            remove_file(path.string());
            log(fmt("the copy of %zu tokens went while its delta was written: the image is written whole",
                    target.live.ids.size()));
            return false;
        }
        Entry n = entries_[i];
        if (n.deltas() == 0) {   // первая дельта: ключи базы - для отката
            n.chain.base_live = n.live;
            n.chain.base_checkpoints = n.checkpoints;
            n.chain.base_digest = n.chain.digest;
        }
        was = (int64_t) n.live.ids.size();
        n.id = job.entry.id;   // новое состояние - новый id: best() прошлого состояния не отдаст это
        n.live = std::move(job.entry.live);
        n.checkpoints = std::move(job.entry.checkpoints);
        n.cvec = job.entry.cvec;
        n.files.push_back(path.string());
        n.bytes += f.n;
        n.chain.table = p.table;
        n.chain.digest = p.digest;
        if (options_.persist && !write_chain_file(n, error)) {
            remove_file(path.string());
            ++stats_.write_failed;
            log(fmt("dropped %zu tokens: the chain's list cannot be written (%s)", tokens, error.c_str()));
            ok = false;
        } else {
            bytes_ += f.n;
            entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
            entries_.push_back(std::move(n));
            ++stats_.written;
            ++stats_.deltas;
            stats_.bytes_written += f.n;
            stats_.delta_bytes += f.n;
            stats_.delta_image_bytes += p.image_bytes;
            stats_.write_ms += ms;
            on_disk = entries_.size();
            disk_bytes = bytes_;
            links = entries_.back().deltas();
        }
    }
    job.image = SavedConversation{};   // the RAM goes back now
    p.parts.clear();
    if (ok)
        log(fmt("wrote delta %zu tokens (+%lld tokens, %.1f MiB of %.0f MiB whole; delta %zu of %zu), %.0f ms; on disk "
                "%zu, %.0f MiB; %zu removed for the budget", tokens, (long long) tokens - was, mib(f.n),
                mib(p.image_bytes), links, options_.max_deltas, ms, on_disk, mib(disk_bytes), made_room));
    return true;
}

// Откат цепочки, в которой лежит `bad_delta`, на её базу: дельты удаляются, ключи - ключи базы, новый id.
bool ConversationDiskTier::cut_back_locked(const std::string& bad_delta, std::string& note) {
    for (auto& e : entries_) {
        if (e.deltas() == 0 ||
            std::find(e.files.begin() + (std::ptrdiff_t) e.chain.base_files, e.files.end(), bad_delta) == e.files.end())
            continue;
        const size_t dropped = e.deltas();
        for (size_t k = e.chain.base_files; k < e.files.size(); ++k) drop_file_locked(e.files[k]);
        e.files.resize(e.chain.base_files);
        bytes_ -= e.bytes - e.chain.base_bytes;
        e.bytes = e.chain.base_bytes;
        e.live = std::move(e.chain.base_live);
        e.checkpoints = std::move(e.chain.base_checkpoints);
        e.chain.base_live = ConversationCheckpoint{};
        e.chain.base_checkpoints.clear();
        e.chain.known = false;   // подписей базы нет: следующая запись этой беседы - целиком
        e.chain.table.reset();
        e.chain.digest = e.chain.base_digest;
        e.id = ++next_id_;
        if (options_.persist) {
            std::string error;
            if (!write_chain_file(e, error)) {   // без описи открытие возьмёт базу по индексу
                if (!e.chain.file.empty()) remove_file(e.chain.file);
                e.chain.file.clear();
            }
        }
        note = fmt("the chain is cut back to its base of %zu tokens (%zu delta%s removed)", e.live.ids.size(), dropped,
                   dropped == 1 ? "" : "s");
        return true;
    }
    return false;
}

namespace {

// Одна дельта поверх `out` (состояние до неё): поля берутся из `out` по ключу, им ставятся новые длины и
// записываются изменившиеся блоки; `dirty` получает номера записанных блоков.  Состояние до дельты сверено по
// отпечатку `prev`; `digest` - отпечаток состояния после неё.
bool apply_delta(const std::string& b, const SessionFileIdentity& id, uint64_t block, uint64_t prev,
                 SavedConversation& out, std::map<uint64_t, std::set<uint32_t>>& dirty, uint64_t& digest,
                 std::string& error) {
    IndexReader in{b};
    if (in.get<uint64_t>() != kDeltaMagic || in.get<uint64_t>() != id.model || in.get<uint64_t>() != id.config) {
        error = "not a delta of this tier (another run, model or settings)";
        return false;
    }
    if (in.get<uint64_t>() != block) {
        error = "a delta of another block size";
        return false;
    }
    if (in.get<uint64_t>() != prev) {
        error = "the delta does not follow the state before it (a foreign delta)";
        return false;
    }
    digest = in.get<uint64_t>();
    const uint64_t skel = in.get<uint64_t>();
    if (!in.ok || in.left() < skel) {
        error = "the delta ends early";
        return false;
    }
    IndexReader sk{in.b.substr(in.at, (size_t) skel)};
    in.at += (size_t) skel;
    std::vector<SavedConversation> stages;
    if (!skeleton_read(sk, stages) || sk.left() != 0) {
        error = "the delta's skeleton is damaged";
        return false;
    }
    SavedConversation img = std::move(stages[0]);
    for (size_t k = 1; k < stages.size(); ++k) img.stage_images.push_back(std::move(stages[k]));
    std::vector<Part> was, now;
    if (!image_parts(out, was) || !image_parts(img, now)) {
        error = "the delta's skeleton does not split into fields";
        return false;
    }
    std::map<uint64_t, Part*> old, fresh;
    for (auto& p : was) old[p.key] = &p;
    for (auto& p : now) fresh[p.key] = &p;
    for (auto& p : now)
        if (const auto it = old.find(p.key); it != old.end()) p.take_from(*it->second);
    const uint32_t n = in.get<uint32_t>();
    if (!in.ok || n != now.size()) {
        error = "the delta lists another set of fields than its skeleton";
        return false;
    }
    std::set<uint64_t> seen;
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t key = in.get<uint64_t>(), len = in.get<uint64_t>();
        const uint32_t nch = in.get<uint32_t>();
        const auto it = fresh.find(key);
        if (!in.ok || it == fresh.end() || !seen.insert(key).second || len > (1ull << 40) || !it->second->resize(len)) {
            error = "the delta's field list is damaged";
            return false;
        }
        Part& p = *it->second;
        auto& d = dirty[key];
        for (uint32_t c = 0; c < nch; ++c) {
            const uint32_t bi = in.get<uint32_t>();
            const uint64_t off = (uint64_t) bi * block;
            if (!in.ok || off >= len || in.left() < std::min(block, len - off)) {
                error = "the delta's blocks are damaged";
                return false;
            }
            const uint64_t blen = std::min(block, len - off);
            const char* src = in.b.data() + in.at;
            p.visit((size_t) off, (size_t) blen, [&](uint8_t* q, size_t cnt, size_t at) {
                std::memcpy(q, src + (at - off), cnt);
                return true;
            });
            in.at += (size_t) blen;
            d.insert(bi);
        }
    }
    if (!in.ok || in.left() != 0) {
        error = "the delta has bytes past its fields";
        return false;
    }
    out = std::move(img);
    return true;
}

// Собранный образ против подписей, с которыми его писали: длины всех полей, хеши блоков, записанных дельтами
// (остальные - байты базы, проверенные её хешем), и отпечаток всего состояния.
bool verify_chain(SavedConversation& out, const ConversationDiskTier::Table& table, uint64_t block, uint64_t digest,
                  const std::map<uint64_t, std::set<uint32_t>>& dirty, std::string& error) {
    std::vector<Part> parts;
    if (!image_parts(out, parts) || parts.size() != table.size()) {
        error = "the assembled image has other fields than its chain was written with";
        return false;
    }
    for (auto& p : parts) {
        const auto f = table.find(p.key);
        if (f == table.end() || f->second.len != p.size()) {
            error = "the assembled image has other fields than its chain was written with";
            return false;
        }
        const auto d = dirty.find(p.key);
        if (d == dirty.end()) continue;
        for (const uint32_t bi : d->second) {
            if (bi >= f->second.blocks.size()) continue;   // поле потом укоротилось
            const uint64_t off = (uint64_t) bi * block;
            if (part_hash(p, (size_t) off, (size_t) std::min<uint64_t>(block, p.size() - off)) != f->second.blocks[bi]) {
                error = "the assembled image differs from the one written";
                return false;
            }
        }
    }
    if (state_digest(skeleton_of(out), table) != digest) {
        error = "the assembled image differs from the one written (its skeleton)";
        return false;
    }
    return true;
}

}  // namespace

bool ConversationDiskTier::read_chain(const Entry& want, SavedConversation& out, std::string& error,
                                      const std::function<bool(uint64_t, std::string&)>& admit, uint64_t& bytes,
                                      size_t& bad, bool& memory) {
    const SessionFileIdentity fid = file_identity();
    SessionReadLimits limits;
    limits.admit = admit;
    for (size_t k = 0; k < want.chain.base_files; ++k) {
        SavedConversation part;
        size_t n = 0;
        SessionStatus status;
        const bool ok = session_file_read(want.files[k], fid, part, n, error, limits, &status);
        bytes += n;
        if (!ok) {
            memory = status.error == SessionError::memory;
            return false;
        }
        if (k == 0) out = std::move(part);
        else out.stage_images.push_back(std::move(part));
    }
    if (want.deltas() == 0) return true;
    uint64_t expect = want.chain.base_digest;
    std::map<uint64_t, std::set<uint32_t>> dirty;
    for (size_t k = want.chain.base_files; k < want.files.size(); ++k) {
        std::error_code ec;
        const uint64_t size = fs::file_size(fs::path(want.files[k]), ec);
        std::string why;
        if (!ec && admit && !admit(2 * size, why)) {   // файл в памяти и прирост полей
            error = why;
            memory = true;
            return false;
        }
        bad = k;
        std::string b;
        if (!read_hashed(want.files[k], b, error)) return false;
        bytes += b.size() + 8;
        uint64_t next = 0;
        if (!apply_delta(b, fid, want.chain.block, expect, out, dirty, next, error)) return false;
        expect = next;
    }
    bad = want.chain.base_files;   // дальше виновника не знать: откат на базу
    if (expect != want.chain.digest) {
        error = "the chain's last delta is not the state its list names";
        return false;
    }
    if (!want.chain.table || !verify_chain(out, *want.chain.table, want.chain.block, want.chain.digest, dirty, error))
        return false;
    bad = SIZE_MAX;
    return true;
}

bool ConversationDiskTier::take(uint64_t id, int64_t tokens, SavedConversation& out, std::string& error,
                                const std::function<bool(uint64_t, std::string&)>& admit) {
    out = SavedConversation{};
    const auto t0 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(m_);
    if (!open_) { error = "the disk tier is off"; return false; }
    // still queued: back from RAM, no disk at all
    for (size_t i = 0; i < queue_.size(); ++i)
        if (queue_[i].entry.id == id) {
            out = std::move(queue_[i].image);
            const uint64_t b = queue_[i].entry.bytes;
            queue_.erase(queue_.begin() + (std::ptrdiff_t) i);
            ++stats_.ram_hits;
            stats_.tokens_back += (uint64_t) std::max<int64_t>(tokens, 0);
            lock.unlock();
            log(fmt("RAM hit: %lld tokens back from the write queue (%.0f MiB, no disk) in %.1f ms",
                    (long long) tokens, mib(b), ms_since(t0)));
            log(summary());
            return true;
        }
    // being written: its file is complete when the write ends
    double waited = 0;
    if (writing_ && writing_entry_.id == id) {
        const auto w0 = std::chrono::steady_clock::now();
        cv_.wait(lock, [&] { return !(writing_ && writing_entry_.id == id); });
        waited = ms_since(w0);
    }
    const size_t index = find_locked(id);
    if (index == entries_.size()) {
        ++stats_.read_failed;
        error = "the entry is gone (its write failed, or the budget removed it)";
        return false;
    }
    const Entry want = entries_[index];
    const std::vector<std::string>& files = want.files;
    // с цепочками запись остаётся на диске - база дельты, когда беседа вернётся - и встаёт последней в очередь
    // вытеснения; её файлы не удаляются, пока их читают
    const bool keep = options_.max_deltas > 0;
    if (keep) {
        Entry moved = std::move(entries_[index]);
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        entries_.push_back(std::move(moved));
        for (const auto& f : files) reading_.insert(f);
    } else {
        remove_locked(index, false);   // the entry goes now; its files are deleted after the read, whatever it gives
    }
    lock.unlock();
    uint64_t bytes = 0;
    size_t bad = SIZE_MAX;
    bool memory = false;
    bool ok = read_chain(want, out, error, admit, bytes, bad, memory);
    if (ok && !holds(out, want.live, want.checkpoints, want.cvec)) {
        ok = false;
        error = "the files hold another conversation than their index";
        if (want.deltas() > 0) bad = want.chain.base_files;
    }
    if (!keep)
        for (const auto& f : files) remove_file(f);
    const double ms = ms_since(t0);
    lock.lock();
    if (keep) {
        for (const auto& f : files) {
            reading_.erase(reading_.find(f));
            if (!reading_.count(f) && doomed_.erase(f)) remove_file(f);
        }
    }
    if (!ok) {
        ++stats_.read_failed;
        std::string note;
        if (keep) {
            if (bad != SIZE_MAX && cut_back_locked(files[bad], note)) {
                ++stats_.chain_cuts;
            } else if (memory) {
                note = "the entry stays on the disk";
            } else {
                for (size_t i = 0; i < entries_.size(); ++i)
                    if (!entries_[i].files.empty() && entries_[i].files[0] == files[0]) {
                        remove_locked(i, true);
                        note = "the entry and its files are removed";
                        break;
                    }
            }
        }
        lock.unlock();
        out = SavedConversation{};
        log(fmt("read of %lld tokens failed after %.0f ms (%s)%s%s", (long long) tokens, ms, error.c_str(),
                note.empty() ? "" : "; ", note.c_str()));
        return false;
    }
    ++stats_.disk_hits;
    stats_.bytes_read += bytes;
    stats_.read_ms += ms - waited;
    stats_.tokens_back += (uint64_t) std::max<int64_t>(tokens, 0);
    lock.unlock();
    const size_t deltas = want.deltas();
    log(fmt("disk hit: %lld tokens, %.0f MiB in %zu file%s%s, %.0f ms (%.0f MB/s)%s", (long long) tokens, mib(bytes),
            files.size(), files.size() == 1 ? "" : "s",
            deltas ? fmt(" (a base and %zu delta%s)", deltas, deltas == 1 ? "" : "s").c_str() : "", ms - waited,
            mbps(bytes, ms - waited), waited > 0 ? fmt(", after %.0f ms waiting for its write", waited).c_str() : ""));
    log(summary());
    return true;
}

bool ConversationDiskTier::pending() const {
    std::lock_guard<std::mutex> lock(m_);
    return !queue_.empty() || writing_;
}

double ConversationDiskTier::wait_idle() {
    std::unique_lock<std::mutex> lock(m_);
    if (queue_.empty() && !writing_) return 0;
    const auto t0 = std::chrono::steady_clock::now();
    const size_t depth = queue_.size() + (writing_ ? 1 : 0);
    cv_.wait(lock, [&] { return queue_.empty() && !writing_; });
    const double ms = ms_since(t0);
    ++stats_.waits;
    stats_.wait_ms += ms;
    lock.unlock();
    log(fmt("waited %.0f ms for %zu queued write%s (their RAM was needed)", ms, depth, depth == 1 ? "" : "s"));
    return ms;
}

size_t ConversationDiskTier::size() const {
    std::lock_guard<std::mutex> lock(m_);
    return entries_.size();
}

uint64_t ConversationDiskTier::bytes() const {
    std::lock_guard<std::mutex> lock(m_);
    return bytes_;
}

ConversationDiskStats ConversationDiskTier::stats() const {
    std::lock_guard<std::mutex> lock(m_);
    return stats_;
}

size_t ConversationDiskTier::clear() {
    std::unique_lock<std::mutex> lock(m_);
    const size_t queued = queue_.size();
    queue_.clear();
    cv_.wait(lock, [&] { return !writing_; });
    const size_t n = queued + entries_.size();
    while (!entries_.empty()) remove_locked(0, true);
    lock.unlock();
    if (n) log(fmt("removed all %zu conversations (#879)", n));
    return n;
}

std::string ConversationDiskTier::summary() const {
    std::lock_guard<std::mutex> lock(m_);
    const auto& s = stats_;
    return fmt("totals: %zu queued, %zu written (%.0f MiB, %.0f MB/s; %zu as deltas: %.0f MiB instead of %.0f MiB whole), "
               "%zu compactions, %zu chains cut back, %zu RAM hits, %zu disk hits (%.0f MiB, %.0f MB/s), %llu prompt "
               "tokens given back, %zu waits for RAM (%.0f ms), %zu removed for the budget, %zu/%zu writes/reads failed; "
               "on disk %zu (%.0f MiB), queue %zu",
               s.queued, s.written, mib(s.bytes_written), mbps(s.bytes_written, s.write_ms), s.deltas,
               mib(s.delta_bytes), mib(s.delta_image_bytes), s.compactions, s.chain_cuts, s.ram_hits, s.disk_hits,
               mib(s.bytes_read), mbps(s.bytes_read, s.read_ms), (unsigned long long) s.tokens_back, s.waits,
               s.wait_ms, s.evicted, s.write_failed, s.read_failed, entries_.size(), mib(bytes_),
               queue_.size() + (writing_ ? 1 : 0));
}

}  // namespace strata::core
