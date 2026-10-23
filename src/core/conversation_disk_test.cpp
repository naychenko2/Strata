// CPU-only tests for the disk tier of the parked conversations (include/strata/core/conversation_disk.hpp).
// Built with -DSTRATA_BUILD_CONVERSATION_TESTS=ON; no CUDA, no model.
#include "strata/core/conversation_disk.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

ConversationBuffer pattern(size_t n, uint8_t seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t c, size_t at) {
        for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 131u + seed);
        return true;
    });
    return b;
}

std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

ConversationCheckpoint checkpoint(size_t tokens, uint8_t seed, int32_t first = 1000) {
    ConversationCheckpoint c;
    for (size_t i = 0; i < tokens; ++i) c.ids.push_back(first + int32_t(i));
    c.gdn = bytes_of(4099, seed);
    c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(301, seed + 2);
    c.dead = bytes_of(64, seed + 3);
    c.block_pos = bytes_of(8, seed + 4);
    c.used = 42 + seed;
    return c;
}

// one stage's image: `tokens` live tokens from `first`, checkpoints at a half and a quarter of them, two K/V layers
SavedConversation image(size_t tokens, uint8_t seed, int32_t first = 1000, int64_t lo = 0, int64_t hi = 48) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = lo; s.layer_hi = hi;
    s.cvec = false;
    s.live = checkpoint(tokens, seed, first);
    s.checkpoints.push_back(checkpoint(tokens / 2, uint8_t(seed + 9), first));    // the deepest first
    s.checkpoints.push_back(checkpoint(tokens / 4, uint8_t(seed + 19), first));
    for (int layer = 0; layer < 2; ++layer) {
        ConversationKv kv;
        kv.format = 2; kv.cells = 1024; kv.heads = 2; kv.head_dim = 256; kv.page_size = 64; kv.pooled_rows = 9;
        kv.idx_dim = 128;
        kv.k = pattern(layer == 0 ? (17u << 20) + 5 : 4096, uint8_t(seed + layer));   // > 16 MiB: several segments
        kv.v = pattern(4096, uint8_t(seed + 10));
        kv.k_scale = pattern(512, uint8_t(seed + 20));
        kv.v_scale = pattern(512, uint8_t(seed + 30));
        kv.pooled = pattern(9 * 128 * 4, uint8_t(seed + 40));
        s.kv.push_back(std::move(kv));
    }
    return s;
}

// a layer split's image: the first stage's with one later stage's
SavedConversation split_image(size_t tokens, uint8_t seed, int32_t first = 1000) {
    SavedConversation s = image(tokens, seed, first, 0, 26);
    s.stage_images.push_back(image(tokens, uint8_t(seed + 50), first, 26, 48));
    return s;
}

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.imgs == b.imgs && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails &&
           a.dead == b.dead && a.block_pos == b.block_pos && a.used == b.used;
}

bool same(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec ||
        !same_checkpoint(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size() ||
        a.stage_images.size() != b.stage_images.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!same_checkpoint(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i) {
        const auto& x = a.kv[i]; const auto& y = b.kv[i];
        if (x.format != y.format || x.cells != y.cells || x.heads != y.heads || x.head_dim != y.head_dim ||
            x.page_size != y.page_size || x.pooled_rows != y.pooled_rows || x.idx_dim != y.idx_dim ||
            !(x.k == y.k) || !(x.v == y.v) || !(x.k_scale == y.k_scale) || !(x.v_scale == y.v_scale) ||
            !(x.pooled == y.pooled)) return false;
    }
    for (size_t i = 0; i < a.stage_images.size(); ++i)
        if (!same(a.stage_images[i], b.stage_images[i])) return false;
    return true;
}

std::vector<int32_t> prompt_of(size_t tokens, int32_t first = 1000) {
    std::vector<int32_t> p;
    for (size_t i = 0; i < tokens; ++i) p.push_back(first + int32_t(i));
    return p;
}

size_t tier_files(const fs::path& dir) {
    size_t n = 0;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().filename().string().find(ConversationDiskTier::file_prefix()) != std::string::npos) ++n;
    return n;
}

void touch(const fs::path& p) { std::ofstream(p, std::ios::binary) << "x"; }

std::vector<std::string> lines;   // the tier's diagnostic lines
std::mutex lines_m;
// deltas = 0: без цепочек (take() убирает запись) - так проверено всё, что было до цепочек
ConversationDiskOptions opts(const fs::path& dir, uint64_t budget, size_t deltas = 0) {
    ConversationDiskOptions o;
    o.dir = dir.string();
    o.budget_bytes = budget;
    o.max_deltas = deltas;
    o.log = [](const std::string& l) { std::lock_guard<std::mutex> g(lines_m); lines.push_back(l); };
    return o;
}
bool logged(const char* what) {
    std::lock_guard<std::mutex> g(lines_m);
    for (const auto& l : lines) if (l.find(what) != std::string::npos) return true;
    return false;
}

// Цепочки: маленький блок, чтобы дельта была видна на образах в десятки МиБ.
ConversationDiskOptions chain_opts(const fs::path& dir, uint64_t budget, size_t deltas = 8) {
    auto o = opts(dir, budget, deltas);
    o.delta_block_bytes = 4096;
    return o;
}

// Следующий ход той же беседы: +add токенов, новое running state, контрольная точка там, где был конец, K/V каждого
// слоя длиннее на kv_add байт (начало не меняется) - в каждой стадии.
SavedConversation turn(const SavedConversation& prev, size_t add, uint8_t seed, size_t kv_add = 3000) {
    SavedConversation s = prev;
    auto step = [&](SavedConversation& x, uint8_t sd) {
        ConversationCheckpoint ck = checkpoint(0, uint8_t(sd + 100));
        ck.ids = x.live.ids;
        ck.imgs = x.live.imgs;
        x.checkpoints.insert(x.checkpoints.begin(), std::move(ck));   // самая глубокая - первой
        const int32_t next = x.live.ids.empty() ? 1000 : x.live.ids.back() + 1;
        for (size_t i = 0; i < add; ++i) x.live.ids.push_back(next + int32_t(i));
        x.live.gdn = bytes_of(x.live.gdn.size(), sd);
        x.live.ple = bytes_of(x.live.ple.size(), uint8_t(sd + 1));
        x.live.tails = bytes_of(x.live.tails.size(), uint8_t(sd + 2));
        x.live.used += 1;
        for (auto& kv : x.kv)
            for (ConversationBuffer* b : {&kv.k, &kv.v}) {
                const size_t was = b->size();
                b->resize(was + kv_add);
                b->visit(was, kv_add, [&](uint8_t* p, size_t c, size_t at) {
                    for (size_t i = 0; i < c; ++i) p[i] = uint8_t((at + i) * 17u + sd);
                    return true;
                });
            }
    };
    step(s, seed);
    for (auto& st : s.stage_images) step(st, uint8_t(seed + 50));
    return s;
}

std::string file_bytes(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Байт в байт: каждая стадия обоих образов - в сессионный файл, файлы равны.
bool same_files(const SavedConversation& a, const SavedConversation& b, const fs::path& dir) {
    if (a.stage_images.size() != b.stage_images.size()) return false;
    for (size_t k = 0; k <= a.stage_images.size(); ++k) {
        const SavedConversation& x = k ? a.stage_images[k - 1] : a;
        const SavedConversation& y = k ? b.stage_images[k - 1] : b;
        size_t n = 0;
        std::string e;
        const fs::path fx = dir / "same-a.ses", fy = dir / "same-b.ses";
        if (!session_file_write(fx.string(), x, {1, 2}, n, e) || !session_file_write(fy.string(), y, {1, 2}, n, e))
            return false;
        const bool eq = file_bytes(fx) == file_bytes(fy);
        fs::remove(fx);
        fs::remove(fy);
        if (!eq) return false;
    }
    return true;
}

std::vector<fs::path> deltas_in(const fs::path& dir) {
    std::vector<fs::path> v;
    for (const auto& f : fs::directory_iterator(dir))
        if (f.path().filename().string().find("-delta.bin") != std::string::npos) v.push_back(f.path());
    std::sort(v.begin(), v.end());
    return v;
}
}  // namespace

int main() {
    const fs::path dir = fs::temp_directory_path() / ("strata_conversation_disk_test_" + std::to_string(std::rand()));
    fs::remove_all(dir);
    const uint64_t GiB = 1ull << 30;
    std::string e;

    {   // off without a folder or a budget
        ConversationDiskTier off(opts("", GiB));
        check(off.open(e) && !off.enabled() && !off.put(image(10, 1)), "no folder: off, not an error");
        ConversationDiskTier off2(opts(dir, 0));
        check(off2.open(e) && !off2.enabled(), "no budget: off");
    }

    {   // start: this tier's leftovers (and their temporary files) go, other files stay
        fs::create_directories(dir);
        touch(dir / (std::string(ConversationDiskTier::file_prefix()) + "0123-1-0.bin"));
        touch(dir / ("." + std::string(ConversationDiskTier::file_prefix()) + "0123-1-0.bin.abcd.tmp"));
        touch(dir / "keep.bin");
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e) && tier.enabled(), "open");
        check(tier_files(dir) == 0 && logged("2 stale files removed"), "open removes the tier's old and temporary files");
        check(fs::exists(dir / "keep.bin"), "open keeps other files");
        fs::remove(dir / "keep.bin");
    }

    {   // background write, match, read back exactly (a layer split: two files; every checkpoint)
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e), "open 2");
        const SavedConversation expect = split_image(600, 1);
        check(tier.put(split_image(600, 1)), "put a split image");
        tier.wait_idle();
        check(tier.size() == 1 && tier_files(dir) == 2 && !tier.pending(), "written: one file per stage");
        check(logged("wrote 600 tokens") && logged("in 2 files"), "the write is logged");
        const auto live = tier.best(prompt_of(700), {}, false);
        check(live.tokens == 600 && live.live && live.id != 0, "best: the live state as a prefix");
        check(!live.queued && live.bytes == tier.bytes() && live.bytes > 0, "best: a written entry, its files' bytes");
        const auto ck = tier.best([] { auto p = prompt_of(300); p.push_back(7); return p; }(), {}, false);
        check(ck.tokens == 300 && !ck.live, "best: the deepest checkpoint as a prefix");
        check(tier.best([] { auto p = prompt_of(150); p.push_back(7); return p; }(), {}, false).tokens == 150,
              "best: an earlier checkpoint is kept too");
        const auto whole = tier.best(prompt_of(600), {}, false);
        check(whole.tokens == 300 && !whole.live, "best: the whole prompt held is no prefix (its last token is read)");
        check(tier.best(prompt_of(700), {}, true).tokens == 0, "best: another control-vector state does not match");
        check(tier.best(prompt_of(700, 5), {}, false).tokens == 0, "best: other tokens do not match");
        SavedConversation back;
        check(tier.take(live.id, live.tokens, back, e), "take");
        check(same(back, expect), "read back exactly: both stages, the live state and every checkpoint");
        check(tier.size() == 0 && tier.bytes() == 0 && tier_files(dir) == 0, "take removes the entry and its files");
        const auto st = tier.stats();
        check(st.queued == 1 && st.written == 1 && st.disk_hits == 1 && st.ram_hits == 0 && st.tokens_back == 600 &&
              st.bytes_read == st.bytes_written && st.bytes_written > (34u << 20), "counters");
        check(logged("disk hit: 600 tokens") && logged("totals:"), "the hit and the totals are logged");
    }

    {   // a queued image comes back from RAM: the writer is held at its first write
        std::promise<void> release;
        std::shared_future<void> gate = release.get_future().share();
        auto o = opts(dir, GiB);
        o.before_write = [gate] { gate.wait(); };
        ConversationDiskTier tier(std::move(o));
        check(tier.open(e), "open 3");
        check(tier.put(image(400, 2)), "put a (the writer takes it and waits)");
        check(tier.put(image(400, 3, 5000)), "put b (queued behind a)");
        check(tier.pending(), "pending while the writer waits");
        const auto b = tier.best(prompt_of(450, 5000), {}, false);
        check(b.tokens == 400 && b.queued && b.bytes > 0, "b matches in the queue");
        SavedConversation back;
        check(tier.take(b.id, b.tokens, back, e) && same(back, image(400, 3, 5000)), "b back from RAM");
        check(tier.stats().ram_hits == 1 && logged("RAM hit: 400 tokens"), "a RAM hit, logged");
        release.set_value();
        tier.wait_idle();
        check(!tier.pending() && tier.size() == 1 && tier_files(dir) == 1, "a written after the release");
        check(tier.best(prompt_of(450, 5000), {}, false).tokens == 0, "b is gone from the tier");
    }

    {   // #879: clear() takes the conversations on the disk, their files too
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e) && tier.put(image(400, 40)), "clear: put a");
        tier.wait_idle();
        check(tier.size() == 1 && tier_files(dir) == 1, "clear: a on the disk");
        check(tier.clear() == 1 && tier.size() == 0 && tier_files(dir) == 0 && logged("removed all 1 conversations"),
              "clear: a and its file gone");
        check(tier.best(prompt_of(450), {}, false).tokens == 0, "clear: nothing matches");
    }

    {   // the budget: the oldest goes for a new one; an image above the budget is refused
        ConversationDiskTier tier(opts(dir, 25ull << 20));   // one ~17 MiB image, not two
        check(tier.open(e), "open 4");
        check(tier.put(image(400, 4)), "first queued");
        tier.wait_idle();
        check(tier.size() == 1, "first fits");
        check(tier.put(image(500, 5, 5000)), "second queued");
        tier.wait_idle();
        check(tier.size() == 1 && tier.stats().evicted == 1, "second evicts the first");
        check(tier.best(prompt_of(450), {}, false).tokens == 0 && tier.best(prompt_of(600, 5000), {}, false).tokens == 500,
              "the newer one stays");
        ConversationDiskTier small(opts(dir, 1ull << 20));
        check(small.open(e) && !small.put(image(400, 6)) && small.size() == 0, "above the budget: refused");
    }
    check(tier_files(dir) == 0, "the tier removes its files when it ends");

    {   // a read the RAM preflight refuses: nothing restored, the files are gone anyway
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e) && tier.put(split_image(400, 10)), "put for a refused read");
        tier.wait_idle();
        const auto m = tier.best(prompt_of(450), {}, false);
        SavedConversation back;
        const bool ok = tier.take(m.id, m.tokens, back, e, [](uint64_t, std::string& why) { why = "no RAM"; return false; });
        check(!ok && back.kv.empty() && back.stage_images.empty() && tier.size() == 0 && tier_files(dir) == 0,
              "refused read: empty image, entry and files gone");
        check(tier.stats().read_failed == 1, "a failed read is counted");
    }

    {   // the RAM cache's evictions go to the disk tier (set_on_evict)
        ConversationDiskTier tier(opts(dir, GiB));
        check(tier.open(e), "open 6");
        ConversationCache cache(GiB, 1);
        cache.set_on_evict([&](SavedConversation&& s) { tier.put(std::move(s)); });
        check(cache.put(image(400, 11)), "park a");
        check(cache.put(image(400, 12, 8000)), "park b: a leaves RAM");
        tier.wait_idle();
        check(cache.size() == 1 && cache.evictions() == 1 && tier.size() == 1, "a is on the disk");
        check(tier.best(prompt_of(450), {}, false).tokens == 400, "a matches on the disk");
        check(cache.best(prompt_of(450, 8000), {}, false).tokens == 400, "b stays in RAM");
    }

    {   // persist: the files outlive the tier, and the next tier with the same identity takes them in
        auto persistent = [&](uint64_t model) {
            auto o = opts(dir, GiB);
            o.persist = true;
            o.identity = {model, 7};
            return o;
        };
        const std::string prefix = ConversationDiskTier::file_prefix();
        {
            ConversationDiskTier tier(persistent(1));
            check(tier.open(e) && tier.persistent(), "persist: open");
            check(tier.put(split_image(500, 20)) && tier.put(image(400, 21, 6000)), "persist: two queued");
        }   // the end waits for the queue and keeps the files
        check(tier_files(dir) == 5 && logged("2 conversations kept on the disk"), "persist: 3 images + 2 indexes kept");
        touch(dir / ("." + prefix + "x-1-0.bin.abcd.tmp"));
        touch(dir / (prefix + "orphan-9-0.bin"));
        {
            ConversationDiskTier tier(persistent(1));
            check(tier.open(e) && tier.size() == 2 && tier_files(dir) == 5, "persist: taken in, leftovers removed");
            check(logged("2 conversations from earlier runs"), "persist: the start is logged");
            const auto m = tier.best(prompt_of(600), {}, false);
            check(m.tokens == 500 && m.live, "persist: matches after a restart");
            SavedConversation back;
            const bool read = tier.take(m.id, m.tokens, back, e);
            if (!read) std::fprintf(stderr, "take: %s\n", e.c_str());
            check(read, "persist: read back after a restart");
            check(same(back, split_image(500, 20)), "persist: the image is exactly the one written");
            check(tier_files(dir) == 2, "persist: its images and index are gone");
        }
        {
            ConversationDiskTier other(persistent(2));   // another model, engine version or settings
            check(other.open(e) && other.size() == 0 && tier_files(dir) == 0 && logged("1 refused"),
                  "persist: another identity's files are refused and removed");
        }
        {
            ConversationDiskTier tier(persistent(3));
            check(tier.open(e) && tier.put(image(300, 22)), "persist: put for a damaged index");
        }
        for (const auto& f : fs::directory_iterator(dir))
            if (f.path().extension() == ".idx") fs::resize_file(f.path(), 20);
        {
            ConversationDiskTier tier(persistent(3));
            check(tier.open(e) && tier.size() == 0 && tier_files(dir) == 0, "persist: a damaged index is refused");
        }
        {
            ConversationDiskTier tier(persistent(4));
            check(tier.open(e) && tier.put(image(400, 23)) && tier.put(image(400, 24, 6000)),
                  "persist: put a and b, the same length");
        }
        std::vector<fs::path> bins;   // a's image first: the names differ only in the id
        for (const auto& f : fs::directory_iterator(dir))
            if (f.path().extension() == ".bin") bins.push_back(f.path());
        std::sort(bins.begin(), bins.end());
        check(bins.size() == 2, "persist: two images");
        fs::copy_file(bins[0], bins[1], fs::copy_options::overwrite_existing);   // b's index now names a's image
        {
            ConversationDiskTier tier(persistent(4));
            check(tier.open(e) && tier.size() == 2, "persist: both taken in");
            const auto m = tier.best(prompt_of(450, 6000), {}, false);
            check(m.tokens == 400, "persist: b matches by its index");
            SavedConversation back;
            check(!tier.take(m.id, m.tokens, back, e) && back.kv.empty() && logged("hold another conversation"),
                  "persist: a's image under b's index is refused, never restored as b");
            check(tier.best(prompt_of(450), {}, false).tokens == 400, "persist: a is still there");
        }
    }

    for (const auto& f : fs::directory_iterator(dir)) fs::remove_all(f.path());

    // ---- цепочки: база + дельты

    {   // дельта пишется и читается; цепочка собирается в тот же образ, что запись целиком (байт в байт)
        ConversationDiskTier tier(chain_opts(dir, GiB));
        check(tier.open(e) && tier.enabled(), "chain: open");
        const SavedConversation t0 = split_image(600, 1);
        check(tier.put(split_image(600, 1)), "chain: put the base");
        tier.wait_idle();
        const auto m0 = tier.best(prompt_of(700), {}, false);
        SavedConversation back;
        check(tier.take(m0.id, m0.tokens, back, e) && same(back, t0), "chain: the base back");
        check(tier.size() == 1 && tier_files(dir) == 2, "chain: a read keeps the entry and its files (the delta's base)");
        const SavedConversation t1 = turn(t0, 50, 2);
        check(tier.put(turn(t0, 50, 2)), "chain: put the next turn");
        tier.wait_idle();
        auto st = tier.stats();
        check(st.written == 2 && st.deltas == 1 && tier.size() == 1 && tier_files(dir) == 3,
              "chain: the next turn is one delta file beside the base");
        check(logged("wrote delta 650 tokens (+50 tokens"), "chain: the delta is logged");
        check(st.delta_bytes > 0 && st.delta_bytes * 50 < st.delta_image_bytes,
              "chain: the delta is a small part of the image");
        check(tier.bytes() > st.delta_bytes && tier.bytes() == st.bytes_written, "chain: the budget counts the chain");
        check(tier.best(prompt_of(640), {}, false).tokens == 600, "chain: the turn's checkpoint matches");
        const auto m1 = tier.best(prompt_of(800), {}, false);
        check(m1.tokens == 650 && m1.live && m1.id != m0.id, "chain: the new state matches, under a new id");
        back = SavedConversation{};
        check(tier.take(m1.id, m1.tokens, back, e), "chain: take the base and its delta");
        check(same(back, t1), "chain: assembled exactly");
        check(same_files(back, t1, dir), "chain: byte for byte the session files of the whole image");
        check(logged("(a base and 1 delta)"), "chain: the hit names its delta");
        const SavedConversation t2 = turn(t1, 40, 3);
        check(tier.put(turn(t1, 40, 3)), "chain: put a third turn");
        tier.wait_idle();
        check(tier.stats().deltas == 2 && tier_files(dir) == 4, "chain: a second delta");
        const auto m2 = tier.best(prompt_of(800), {}, false);
        back = SavedConversation{};
        check(m2.tokens == 690 && tier.take(m2.id, m2.tokens, back, e) && same(back, t2) && same_files(back, t2, dir),
              "chain: a base and two deltas, byte for byte");
        check(logged("totals:") && logged("as deltas:"), "chain: the totals count the deltas");
    }
    check(tier_files(dir) == 0, "chain: the tier removes its chains when it ends");

    {   // уплотнение по числу дельт: третья запись при пределе 2 переписывает цепочку целиком
        ConversationDiskTier tier(chain_opts(dir, GiB, 2));
        check(tier.open(e), "compaction: open");
        SavedConversation t = image(600, 5);
        check(tier.put(SavedConversation(t)), "compaction: base");
        for (int i = 0; i < 3; ++i) {
            t = turn(t, 30, uint8_t(10 + i));
            check(tier.put(SavedConversation(t)), "compaction: a turn");
            tier.wait_idle();
        }
        const auto st = tier.stats();
        check(st.deltas == 2 && st.compactions == 1 && tier.size() == 1 && tier_files(dir) == 1,
              "compaction: two deltas, then the chain is one whole image again");
        check(logged("compaction: replaces its chain of 660 tokens and 2 deltas (the chain holds 2 deltas)"),
              "compaction: logged with its reason");
        const auto m = tier.best(prompt_of(800), {}, false);
        SavedConversation back;
        check(m.tokens == 690 && tier.take(m.id, m.tokens, back, e) && same(back, t), "compaction: the image back");
        // уплотнение по объёму: ход, переписавший весь K/V (и длиннее его), тяжелее базы
        const SavedConversation heavy = turn(image(700, 77), 0, 78, 200000);
        check(tier.put(SavedConversation(heavy)), "compaction: a turn that rewrites everything");
        tier.wait_idle();
        check(tier.stats().compactions == 2 && tier.stats().deltas == 2 && logged("would outweigh the base"),
              "compaction: deltas heavier than the base are not written");
        const auto h = tier.best(prompt_of(800), {}, false);
        back = SavedConversation{};
        check(h.tokens == 700 && tier.take(h.id, h.tokens, back, e) && same(back, heavy) && tier_files(dir) == 1,
              "compaction: the heavy turn whole, the old chain gone");
    }

    {   // вытеснение - цепочкой целиком; своя дельта свою базу не вытесняет
        ConversationDiskTier tier(chain_opts(dir, 25ull << 20));   // одна ~17 MiB цепочка, не две
        check(tier.open(e), "chain eviction: open");
        const SavedConversation a0 = image(400, 30);
        check(tier.put(SavedConversation(a0)), "chain eviction: a");
        check(tier.put(turn(a0, 20, 31)), "chain eviction: a's next turn");
        tier.wait_idle();
        check(tier.size() == 1 && tier.stats().deltas == 1 && tier.stats().evicted == 0 && tier_files(dir) == 2,
              "chain eviction: the delta fits beside its base");
        check(tier.put(image(400, 32, 5000)), "chain eviction: b");
        tier.wait_idle();
        check(tier.size() == 1 && tier.stats().evicted == 1 && tier_files(dir) == 1 && deltas_in(dir).empty(),
              "chain eviction: b evicts a's base and delta together");
        check(tier.best(prompt_of(500), {}, false).tokens == 0 && tier.best(prompt_of(500, 5000), {}, false).tokens == 400,
              "chain eviction: a is gone, b stays");
    }

    {   // битая дельта: take() не удался, цепочка откатилась на базу, база читается
        ConversationDiskTier tier(chain_opts(dir, GiB));
        check(tier.open(e), "cut: open");
        const SavedConversation t0 = split_image(600, 40);
        check(tier.put(SavedConversation(t0)) && tier.put(turn(t0, 50, 41)), "cut: a base and a delta");
        tier.wait_idle();
        const auto d = deltas_in(dir);
        check(d.size() == 1 && tier.stats().deltas == 1, "cut: one delta file");
        {
            std::fstream f(d[0], std::ios::binary | std::ios::in | std::ios::out);
            f.seekp((std::streamoff) fs::file_size(d[0]) / 2);
            const char x = 0x5a;
            f.write(&x, 1);
        }
        const auto m = tier.best(prompt_of(800), {}, false);
        SavedConversation back;
        check(m.tokens == 650 && !tier.take(m.id, m.tokens, back, e) && back.kv.empty(), "cut: the damaged chain fails");
        check(logged("is damaged") && logged("cut back to its base of 600 tokens (1 delta removed)"), "cut: logged");
        const auto st = tier.stats();
        check(st.chain_cuts == 1 && st.read_failed == 1 && tier.size() == 1 && tier_files(dir) == 2 &&
              deltas_in(dir).empty(), "cut: the delta is gone, the base stays");
        const auto b = tier.best(prompt_of(800), {}, false);
        check(b.tokens == 600 && b.live && b.id != m.id, "cut: the base matches again");
        check(tier.take(b.id, b.tokens, back, e) && same(back, t0), "cut: the base is read back");
    }

    {   // чужая дельта (цела, но от другой цепочки): по сцепке отпечатков - откат на базу
        ConversationDiskTier tier(chain_opts(dir, GiB));
        check(tier.open(e), "foreign: open");
        const SavedConversation a0 = image(600, 45);
        check(tier.put(SavedConversation(a0)) && tier.put(turn(a0, 50, 46)), "foreign: chain a");
        tier.wait_idle();
        const auto da = deltas_in(dir);
        const SavedConversation u0 = image(500, 47, 7000);
        check(tier.put(SavedConversation(u0)) && tier.put(turn(u0, 10, 48)), "foreign: chain u");
        tier.wait_idle();
        auto du = deltas_in(dir);
        du.erase(std::remove(du.begin(), du.end(), da.at(0)), du.end());
        check(da.size() == 1 && du.size() == 1 && tier.stats().deltas == 2, "foreign: two chains with a delta each");
        fs::copy_file(du[0], da[0], fs::copy_options::overwrite_existing);   // дельта u на месте дельты a
        const auto m = tier.best(prompt_of(800), {}, false);
        SavedConversation back;
        check(m.tokens == 650 && !tier.take(m.id, m.tokens, back, e) && logged("a foreign delta"),
              "foreign: refused by its link");
        check(tier.stats().chain_cuts == 1 && tier.best(prompt_of(800), {}, false).tokens == 600,
              "foreign: chain a is back at its base");
        const auto mu = tier.best(prompt_of(600, 7000), {}, false);
        check(mu.tokens == 510 && tier.take(mu.id, mu.tokens, back, e) && same(back, turn(u0, 10, 48)),
              "foreign: chain u is untouched");
    }

    {   // persist: цепочка переживает рестарт и растёт дальше; движок без цепочек видит базу
        auto persistent = [&](size_t deltas) {
            auto o = chain_opts(dir, GiB, deltas);
            o.persist = true;
            o.identity = {9, 7};
            return o;
        };
        const SavedConversation t0 = split_image(600, 60);
        const SavedConversation t1 = turn(t0, 50, 61);
        const SavedConversation t2 = turn(t1, 40, 62);
        {
            ConversationDiskTier tier(persistent(8));
            check(tier.open(e) && tier.put(SavedConversation(t0)) && tier.put(SavedConversation(t1)),
                  "chain persist: a base and a delta");
        }
        check(tier_files(dir) == 5 && deltas_in(dir).size() == 1, "chain persist: 2 images, a delta, index and list kept");
        {
            ConversationDiskTier tier(persistent(8));
            check(tier.open(e) && tier.size() == 1 && logged("1 with deltas"), "chain persist: taken in as a chain");
            const auto m = tier.best(prompt_of(800), {}, false);
            SavedConversation back;
            check(m.tokens == 650 && tier.take(m.id, m.tokens, back, e) && same(back, t1),
                  "chain persist: the chain read back after a restart");
            check(tier.put(SavedConversation(t2)), "chain persist: the next turn");
            tier.wait_idle();
            check(tier.stats().deltas == 1 && deltas_in(dir).size() == 2,
                  "chain persist: the field table outlived the restart: a delta again");
        }
        {
            ConversationDiskTier tier(persistent(8));
            check(tier.open(e) && tier.size() == 1, "chain persist: a second restart");
            const auto m = tier.best(prompt_of(800), {}, false);
            SavedConversation back;
            check(m.tokens == 690 && tier.take(m.id, m.tokens, back, e) && same(back, t2) && same_files(back, t2, dir),
                  "chain persist: a base and two deltas after two restarts, byte for byte");
        }
        {   // откат на движок без цепочек: индекс v1 описывает базу - она поднимается, дельты уходят
            ConversationDiskTier tier(persistent(0));
            check(tier.open(e) && tier.size() == 1 && deltas_in(dir).empty() && tier_files(dir) == 3,
                  "chain persist: without chains the base is taken in, the deltas and the list removed");
            const auto m = tier.best(prompt_of(800), {}, false);
            SavedConversation back;
            check(m.tokens == 600 && tier.take(m.id, m.tokens, back, e) && same(back, t0),
                  "chain persist: the base read back by a tier without chains");
        }
        check(tier_files(dir) == 0, "chain persist: taken without chains: gone");
        {   // файлы без цепочек (как их писал прежний движок) читаются базой без дельт; следующий ход - целиком
            {
                ConversationDiskTier tier(persistent(0));
                check(tier.open(e) && tier.put(SavedConversation(t0)), "old files: written without chains");
            }
            ConversationDiskTier tier(persistent(8));
            check(tier.open(e) && tier.size() == 1, "old files: taken in");
            const auto m = tier.best(prompt_of(800), {}, false);
            SavedConversation back;
            check(m.tokens == 600 && tier.take(m.id, m.tokens, back, e) && same(back, t0), "old files: read back");
            check(tier.put(SavedConversation(t1)), "old files: the next turn");
            tier.wait_idle();
            check(tier.stats().deltas == 0 && logged("(the copy on the disk has no field table)") && tier.size() == 1,
                  "old files: the next turn replaces the old copy whole");
            check(tier.put(SavedConversation(t2)), "old files: one more turn");
            tier.wait_idle();
            check(tier.stats().deltas == 1, "old files: from then on deltas");
        }
    }

    {   // RAM не хватило на чтение: с цепочками запись остаётся на диске
        ConversationDiskTier tier(chain_opts(dir, GiB));
        check(tier.open(e) && tier.put(image(400, 70)), "chain refused read: put");
        tier.wait_idle();
        const auto m = tier.best(prompt_of(450), {}, false);
        SavedConversation back;
        check(!tier.take(m.id, m.tokens, back, e, [](uint64_t, std::string& why) { why = "no RAM"; return false; }) &&
              tier.size() == 1 && tier_files(dir) == 1 && logged("the entry stays on the disk"),
              "chain refused read: the entry stays");
    }

    fs::remove_all(dir);
    std::printf("conversation_disk_test: %d checks passed\n", checks);
    return 0;
}
