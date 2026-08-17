/* ============================================================
   Cerita — story reader (iOS style)
   Vanilla JS, no dependencies.

   Stories persist to the backend (/api/stories) when available and
   fall back to localStorage (offline) so the app always works.
   ============================================================ */

"use strict";

/* ---------------- Data & constants ---------------- */

const STORAGE_KEY = "cerita:stories";
const THEME_KEY = "cerita:theme";
const SEEDED_KEY = "cerita:seeded";
const API_URL = "/api/stories";

const CATEGORIES = [
  { key: "fiksi", label: "Fiksi", from: "#007AFF", to: "#5AC8FA" },
  { key: "motivasi", label: "Motivasi", from: "#FF9500", to: "#FFCC00" },
  { key: "horor", label: "Horor", from: "#5856D6", to: "#AF52DE" },
  { key: "romansa", label: "Romansa", from: "#FF2D55", to: "#FF6482" },
  { key: "petualangan", label: "Petualangan", from: "#30B158", to: "#30D158" },
  { key: "misteri", label: "Misteri", from: "#3A3A3C", to: "#6E6E73" },
  { key: "komedi", label: "Komedi", from: "#FF3B30", to: "#FF9500" },
];

const SEED_STORIES = [
  {
    id: "seed-1",
    title: "Langit Senja di Ujung Pelabuhan",
    author: "Nadia Prameswari",
    category: "romansa",
    createdAt: Date.now() - 1000 * 60 * 60 * 5,
    reads: 132,
    favorite: false,
    content:
      "Hujan baru saja reda ketika ia melihatnya untuk pertama kali, berdiri di ujung dermaga dengan payung yang masih meneteskan air.\n\nKota kecil ini selalu terasa sunyi menjelang magrib. Kapal-kapal tua bergoyang pelan mengikuti riak, dan langit perlahan berubah dari jingga menjadi ungu tua. Ia tidak pernah menyangka, pertemuan singkat sore itu akan menjadi awal dari surat-surat yang tak pernah dikirim.\n\nBeberapa tahun kemudian, saat kembali ke pelabuhan yang sama, ia menemukan tumpukan surat itu masih tersimpan rapi di laci meja kayu dekat jendela. Semuanya bercerita tentang senja, tentang dermaga, dan tentang seseorang yang tak pernah sempat ia sapa namanya.",
  },
  {
    id: "seed-2",
    title: "Perjalanan Menembus Kabut",
    author: "Bima Ardiansyah",
    category: "petualangan",
    createdAt: Date.now() - 1000 * 60 * 60 * 26,
    reads: 98,
    favorite: false,
    content:
      "Kompas tua di tangannya menunjuk ke arah yang tidak masuk akal — utara, padahal mereka sedang berjalan menuju selatan.\n\nKabut pagi menelan jejak jalan setapak di depan mereka. Hanya suara burung-burung asing dan derit ransel yang menemani. Tim yang berjumlah lima orang itu telah meninggalkan desa terakhir sejak dua hari lalu, membawa peta warisan kakek yang konon menuntun ke air terjun tersembunyi.\n\nDi balik kabut, mereka menemukan sesuatu yang lebih berharga daripada air terjun itu sendiri: keberanian untuk terus melangkah ketika semua tanda arah seakan berkata untuk berbalik.",
  },
  {
    id: "seed-3",
    title: "Rumah di Ujung Gang",
    author: "Clara Wulandari",
    category: "horor",
    createdAt: Date.now() - 1000 * 60 * 60 * 60 * 3,
    reads: 214,
    favorite: false,
    content:
      "Rumah itu selalu kosong, kata orang-orang. Tetapi setiap malam, lampu di lantai dua menyala tepat pukul tiga.\n\nTidak ada yang berani lewat gang itu setelah gelap. Anak-anak mengarang cerita tentang bayangan yang berdiri di balik jendela, menatap siapa pun yang cukup berani membalas tatapannya. Saya, tentu saja, tidak percaya.\n\nSampai suatu malam, pulang terlalu larut karena hujan, saya melihat lampu itu menyala — dan di balik tirai tipisnya, ada sesosok bayangan yang melambaikan tangan ke arah saya.",
  },
  {
    id: "seed-4",
    title: "Menanam Harapan di Tanah Kering",
    author: "Raka Firmansyah",
    category: "motivasi",
    createdAt: Date.now() - 1000 * 60 * 60 * 24 * 8,
    reads: 76,
    favorite: false,
    content:
      "Musim kemarau tahun itu lebih panjang dari biasanya. Tanah di belakang rumah retak-retak, dan sumur pun mulai mengering.\n\nNamun kakek tetap menyirami sepetak kecil kebunnya setiap subuh, dengan air yang ia kumpulkan setetes demi setetes dari sisa cucian. Tetangga menertawakannya. Untuk apa merawat tanah yang sekeras batu?\n\nMusim hujan datang lebih awal pada akhirnya, dan kebun kakek menjadi yang pertama menghijau di antara pekarangan-pekarangan yang gersang. Dari kakek saya belajar: harapan tidak tumbuh dari tanah yang basah, tetapi dari tangan yang tidak pernah berhenti menyiram.",
  },
  {
    id: "seed-5",
    title: "Pencuri Bulan Purnama",
    author: "Sari Melati",
    category: "misteri",
    createdAt: Date.now() - 1000 * 60 * 60 * 24 * 2,
    reads: 61,
    favorite: false,
    content:
      "Setiap bulan purnama, lukisan bulan di galeri kota selalu hilang tepat tengah malam — lalu kembali sebelum fajar dengan satu bintang tambahan di sudutnya.\n\nPolisi tidak pernah menemukan siapa pelakunya. Tidak ada kamera yang merekam apa pun, tidak ada pintu yang rusak. Warga mulai menyebutnya si pencuri bulan purnama, dan sebagian bahkan menantikan kejadian itu setiap bulannya.\n\nSaya menemukan jawabannya secara tidak sengaja: seorang pelukis tua yang tinggal di loteng galeri, yang setiap bulan purnama hanya ingin menambahkan sedikit cahaya pada lukisan yang ia buat untuk mendiang istrinya.",
  },
];

/* ---------------- State ---------------- */

let stories = [];
let currentFilter = "semua";
let searchTerm = "";
let editingId = null;
let activeStoryId = null;
let serverAvailable = false;
let deletedSnapshot = null; // { story, index } for undo

/* ---------------- DOM ---------------- */

const $ = (sel) => document.querySelector(sel);

const el = {
  list: $("#storyList"),
  empty: $("#emptyState"),
  listMeta: $("#listMeta"),
  search: $("#searchInput"),
  searchClear: $("#searchClear"),
  themeSwitch: $("#themeSwitch"),
  syncPill: $("#syncPill"),
  scrim: $("#scrim"),
  readSheet: $("#readSheet"),
  readScroll: $("#readScroll"),
  readProgressBar: $("#readProgressBar"),
  addSheet: $("#addSheet"),
  storyForm: $("#storyForm"),
  titleInput: $("#titleInput"),
  authorInput: $("#authorInput"),
  contentInput: $("#contentInput"),
  charCount: $("#charCount"),
  categoryGrid: $("#categoryGrid"),
  addSheetTitle: $("#addSheetTitle"),
  saveButton: $("#saveButton"),
  cancelButton: $("#cancelButton"),
  deleteButton: $("#deleteButton"),
  editButton: $("#editButton"),
  shareButton: $("#shareButton"),
  favoriteButton: $("#favoriteButton"),
  favoriteLabel: $("#favoriteLabel"),
  closeReadButton: $("#closeReadButton"),
  alertBackdrop: $("#alertBackdrop"),
  alertTitle: $("#alertTitle"),
  alertMessage: $("#alertMessage"),
  alertCancel: $("#alertCancel"),
  alertConfirm: $("#alertConfirm"),
};

/* ---------------- Persistence ---------------- */

function loadLocal() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return parsed;
    }
  } catch (_) {
    /* ignore */
  }
  return [...SEED_STORIES];
}

function saveLocal() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(stories));
  } catch (_) {
    /* storage full / unavailable */
  }
}

async function loadServer() {
  const res = await fetch(API_URL, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error("server error " + res.status);
  const data = await res.json();
  if (!Array.isArray(data)) throw new Error("bad payload");
  return data;
}

async function saveServer() {
  const res = await fetch(API_URL, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(stories),
  });
  if (!res.ok) throw new Error("server error " + res.status);
}

let serverSaveTimer = null;
function scheduleServerSave() {
  clearTimeout(serverSaveTimer);
  serverSaveTimer = setTimeout(async () => {
    try {
      await saveServer();
    } catch (_) {
      serverAvailable = false;
      showOffline();
    }
  }, 250);
}

function persist() {
  saveLocal();
  if (serverAvailable) scheduleServerSave();
}

function showOffline() {
  el.syncPill.hidden = false;
}

async function initData() {
  try {
    const serverStories = await loadServer();
    serverAvailable = true;
    el.syncPill.hidden = true;

    if (serverStories.length === 0 && !localStorage.getItem(SEEDED_KEY)) {
      // First launch on a fresh backend — seed it.
      stories = [...SEED_STORIES];
      localStorage.setItem(SEEDED_KEY, "1");
      try {
        await saveServer();
      } catch (_) {
        /* fall through; local copy still works */
      }
    } else {
      stories = serverStories;
    }
  } catch (_) {
    serverAvailable = false;
    stories = loadLocal();
    showOffline();
  }
}

/* ---------------- Helpers ---------------- */

function uid() {
  return (
    (crypto.randomUUID && crypto.randomUUID()) ||
    "id-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8)
  );
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function categoryByKey(key) {
  return CATEGORIES.find((c) => c.key === key) || CATEGORIES[0];
}

function gradient(categoryKey) {
  const c = categoryByKey(categoryKey);
  return `linear-gradient(135deg, ${c.from}, ${c.to})`;
}

function excerpt(text, length = 160) {
  const plain = String(text || "").replace(/\s+/g, " ").trim();
  return plain.length > length ? plain.slice(0, length).trimEnd() + "…" : plain;
}

function readTime(text) {
  const words = String(text || "").trim().split(/\s+/).filter(Boolean).length;
  return `${Math.max(1, Math.round(words / 200))} mnt baca`;
}

function formatDate(ts) {
  const now = Date.now();
  const diff = now - ts;
  const min = Math.floor(diff / 60000);
  if (min < 1) return "Baru saja";
  if (min < 60) return `${min} mnt lalu`;
  const hrs = Math.floor(min / 60);
  if (hrs < 24) return `${hrs} jam lalu`;
  const days = Math.floor(hrs / 24);
  if (days === 1) return "Kemarin";
  if (days < 7) return `${days} hari lalu`;
  return new Date(ts).toLocaleDateString("id-ID", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

function formatViews(n) {
  const num = Number(n) || 0;
  if (num >= 1000) return (num / 1000).toFixed(1).replace(".0", "") + "rb";
  return String(num);
}

const HEART_SVG = `<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/></svg>`;

/* ---------------- Toast (with optional action) ---------------- */

function toast(message, actionLabel, actionFn) {
  let node = document.querySelector(".toast");
  if (!node) {
    node = document.createElement("div");
    node.className = "toast";
    document.body.appendChild(node);
  }
  node.innerHTML = "";
  const span = document.createElement("span");
  span.textContent = message;
  node.appendChild(span);

  if (actionLabel && typeof actionFn === "function") {
    const btn = document.createElement("button");
    btn.className = "toast-action";
    btn.textContent = actionLabel;
    btn.addEventListener("click", () => {
      actionFn();
      hideToast(node);
    });
    node.appendChild(btn);
  }

  node.classList.add("is-visible");
  clearTimeout(node._timer);
  node._timer = setTimeout(() => hideToast(node), 3600);
}

function hideToast(node) {
  node.classList.remove("is-visible");
}

/* ---------------- Theme ---------------- */

function systemPrefersDark() {
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function applyTheme(dark, persistFlag) {
  const theme = dark ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", theme);
  el.themeSwitch.checked = dark;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", dark ? "#000000" : "#F2F2F7");
  if (persistFlag) localStorage.setItem(THEME_KEY, theme);
}

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  const dark = saved ? saved === "dark" : systemPrefersDark();
  applyTheme(dark, false);
  el.themeSwitch.addEventListener("change", () => applyTheme(el.themeSwitch.checked, true));
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    if (!localStorage.getItem(THEME_KEY)) applyTheme(e.matches, false);
  });
}

/* ---------------- Rendering ---------------- */

function filteredStories() {
  const term = searchTerm.trim().toLowerCase();
  let result = stories.filter((s) => {
    if (term) {
      const cat = categoryByKey(s.category).label.toLowerCase();
      const hay = `${s.title} ${s.author} ${cat} ${s.content}`.toLowerCase();
      if (!hay.includes(term)) return false;
    }
    if (currentFilter === "terbaru") {
      return Date.now() - s.createdAt < 1000 * 60 * 60 * 24 * 7;
    }
    if (currentFilter === "favorit") {
      return !!s.favorite;
    }
    return true;
  });

  if (currentFilter === "populer") {
    result = result.slice().sort((a, b) => b.reads - a.reads);
  } else {
    result = result.slice().sort((a, b) => b.createdAt - a.createdAt);
  }
  return result;
}

function render() {
  const list = filteredStories();
  el.list.innerHTML = "";

  el.listMeta.hidden = list.length === 0;
  el.listMeta.textContent = `${list.length} cerita${searchTerm.trim() ? " ditemukan" : ""}`;

  if (list.length === 0) {
    el.list.hidden = true;
    el.empty.hidden = false;
    const emptyTitle = el.empty.querySelector(".empty-title");
    const emptyText = el.empty.querySelector(".empty-text");
    const emptyBtn = el.empty.querySelector("#emptyAddButton");

    if (searchTerm.trim()) {
      emptyTitle.textContent = "Tidak ditemukan";
      emptyText.textContent = `Tidak ada cerita yang cocok dengan "${searchTerm.trim()}".`;
      emptyBtn.hidden = true;
    } else if (stories.length === 0) {
      emptyTitle.textContent = "Belum ada cerita";
      emptyText.textContent = "Mulai dengan menulis cerita pertamamu.";
      emptyBtn.hidden = false;
    } else if (currentFilter === "favorit") {
      emptyTitle.textContent = "Belum ada favorit";
      emptyText.textContent = "Ketuk ikon hati untuk menyimpan cerita favoritmu.";
      emptyBtn.hidden = true;
    } else if (currentFilter === "terbaru") {
      emptyTitle.textContent = "Tidak ada cerita baru";
      emptyText.textContent = "Belum ada cerita yang ditulis dalam 7 hari terakhir.";
      emptyBtn.hidden = true;
    } else {
      emptyTitle.textContent = "Kosong";
      emptyText.textContent = "Tidak ada cerita pada filter ini.";
      emptyBtn.hidden = true;
    }
    return;
  }

  el.list.hidden = false;
  el.empty.hidden = true;

  list.forEach((story, index) => {
    const cat = categoryByKey(story.category);
    const card = document.createElement("article");
    card.className = "story-card";
    card.style.animationDelay = `${Math.min(index * 45, 360)}ms`;
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-label", `Baca cerita ${story.title}`);
    card.innerHTML = `
      <div class="story-cover" style="background:${gradient(story.category)}">${escapeHtml(
      (story.title[0] || "?").toUpperCase()
    )}</div>
      <div class="story-main">
        <div class="story-topline">
          <span class="pill">${escapeHtml(cat.label)}</span>
          <span class="story-date">${escapeHtml(formatDate(story.createdAt))}</span>
        </div>
        <h3 class="story-title">${escapeHtml(story.title)}</h3>
        <p class="story-excerpt">${escapeHtml(excerpt(story.content))}</p>
        <div class="story-foot">
          <div class="story-foot-left">
            <span>
              <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.6-6 8-6s8 2 8 6"/></svg>
              ${escapeHtml(story.author)}
            </span>
            <span>
              <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
              ${escapeHtml(readTime(story.content))}
            </span>
            <span>
              <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7z"/><circle cx="12" cy="12" r="3"/></svg>
              ${formatViews(story.reads)}
            </span>
          </div>
          <button class="card-fav${story.favorite ? " is-favorited" : ""}" aria-label="${
      story.favorite ? "Hapus dari favorit" : "Tambah ke favorit"
    }" data-fav-id="${escapeHtml(story.id)}">${HEART_SVG}</button>
        </div>
      </div>`;

    card.addEventListener("click", () => openRead(story.id));
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        openRead(story.id);
      }
    });

    const favBtn = card.querySelector(".card-fav");
    favBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleFavorite(story.id);
    });

    el.list.appendChild(card);
  });
}

/* ---------------- Sheets (open / close) ---------------- */

function openSheet(sheet) {
  sheet.hidden = false;
  el.scrim.hidden = false;
  document.body.style.overflow = "hidden";
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      el.scrim.classList.add("is-visible");
      sheet.classList.add("is-open");
    });
  });
}

function closeSheet(sheet) {
  el.scrim.classList.remove("is-visible");
  sheet.classList.remove("is-open");
  document.body.style.overflow = "";
  setTimeout(() => {
    if (!sheet.classList.contains("is-open")) {
      sheet.hidden = true;
      if (el.readSheet.hidden && el.addSheet.hidden) el.scrim.hidden = true;
    }
  }, 320);
}

function closeAllSheets() {
  if (!el.readSheet.hidden) closeSheet(el.readSheet);
  if (!el.addSheet.hidden) closeSheet(el.addSheet);
}

/* ---------------- Read sheet ---------------- */

function openRead(id) {
  const story = stories.find((s) => s.id === id);
  if (!story) return;
  activeStoryId = id;

  $("#readCover").style.background = gradient(story.category);
  $("#readCover").textContent = story.title[0] ? story.title[0].toUpperCase() : "?";
  $("#readCategory").textContent = categoryByKey(story.category).label;
  $("#readTitle").textContent = story.title;
  $("#readAuthor").textContent = story.author;
  $("#readDate").textContent = formatDate(story.createdAt);
  $("#readTime").textContent = readTime(story.content);
  $("#readViews").textContent = `${formatViews(story.reads)} dibaca`;
  $("#readText").innerHTML = String(story.content)
    .split(/\n+/)
    .filter((p) => p.trim())
    .map((p) => `<p>${escapeHtml(p.trim())}</p>`)
    .join("");

  updateFavoriteButton(story);

  story.reads = (story.reads || 0) + 1;
  persist();

  openSheet(el.readSheet);
  el.readScroll.scrollTop = 0;
  updateProgress();
  render();
}

function updateFavoriteButton(story) {
  const fav = !!story.favorite;
  el.favoriteButton.classList.toggle("is-favorited", fav);
  el.favoriteLabel.textContent = fav ? "Disukai" : "Suka";
}

function toggleFavorite(id) {
  const story = stories.find((s) => s.id === id);
  if (!story) return;
  story.favorite = !story.favorite;
  persist();
  updateFavoriteButton(story);
  render();
  if (story.favorite) toast("Ditambahkan ke favorit.");
}

function updateProgress() {
  const sc = el.readScroll;
  const max = sc.scrollHeight - sc.clientHeight;
  const p = max > 0 ? sc.scrollTop / max : 0;
  el.readProgressBar.style.width = (p * 100).toFixed(1) + "%";
}

async function shareStory() {
  const story = stories.find((s) => s.id === activeStoryId);
  if (!story) return;
  const text = `“${story.title}” — oleh ${story.author}\n\n${excerpt(story.content, 240)}`;
  const url = location.href.split("#")[0];

  if (navigator.share) {
    try {
      await navigator.share({ title: story.title, text, url });
      return;
    } catch (e) {
      if (e && e.name === "AbortError") return;
    }
  }

  try {
    await navigator.clipboard.writeText(
      `“${story.title}” — ${story.author}\n\n${story.content}`
    );
    toast("Teks cerita disalin ke papan klip.");
  } catch (_) {
    toast("Tidak dapat membagikan cerita.");
  }
}

/* ---------------- Add / edit sheet ---------------- */

function buildCategoryChips(selectedKey) {
  el.categoryGrid.innerHTML = "";
  CATEGORIES.forEach((cat) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "category-chip";
    chip.dataset.key = cat.key;
    chip.textContent = cat.label;
    if (cat.key === selectedKey) chip.classList.add("is-selected");
    chip.addEventListener("click", () => {
      el.categoryGrid.querySelectorAll(".category-chip").forEach((c) =>
        c.classList.remove("is-selected")
      );
      chip.classList.add("is-selected");
    });
    el.categoryGrid.appendChild(chip);
  });
}

function selectedCategory() {
  const sel = el.categoryGrid.querySelector(".category-chip.is-selected");
  return sel ? sel.dataset.key : "fiksi";
}

function clearInvalid() {
  [el.titleInput, el.authorInput, el.contentInput].forEach((i) =>
    i.classList.remove("is-invalid")
  );
}

function openAdd() {
  editingId = null;
  el.addSheetTitle.textContent = "Tulis Cerita";
  el.storyForm.reset();
  buildCategoryChips("fiksi");
  updateCharCount();
  clearInvalid();
  openSheet(el.addSheet);
  setTimeout(() => el.titleInput.focus(), 300);
}

function openEdit(id) {
  const story = stories.find((s) => s.id === id);
  if (!story) return;
  editingId = id;
  el.addSheetTitle.textContent = "Edit Cerita";
  el.titleInput.value = story.title;
  el.authorInput.value = story.author;
  el.contentInput.value = story.content;
  buildCategoryChips(story.category);
  updateCharCount();
  clearInvalid();
  closeSheet(el.readSheet);
  openSheet(el.addSheet);
  setTimeout(() => el.titleInput.focus(), 300);
}

function updateCharCount() {
  el.charCount.textContent = el.contentInput.value.length;
}

function validate() {
  let valid = true;
  [el.titleInput, el.authorInput, el.contentInput].forEach((i) => {
    const ok = i.value.trim().length > 0;
    i.classList.toggle("is-invalid", !ok);
    if (!ok) valid = false;
  });
  return valid;
}

function handleSave() {
  if (!validate()) {
    toast("Lengkapi judul, penulis, dan isi cerita.");
    return;
  }

  const existing = editingId ? stories.find((s) => s.id === editingId) : null;
  const data = {
    id: editingId || uid(),
    title: el.titleInput.value.trim(),
    author: el.authorInput.value.trim(),
    category: selectedCategory(),
    content: el.contentInput.value.trim(),
    createdAt: existing ? existing.createdAt : Date.now(),
    reads: existing ? existing.reads || 0 : 0,
    favorite: existing ? !!existing.favorite : false,
  };

  if (editingId) {
    stories = stories.map((s) => (s.id === editingId ? data : s));
  } else {
    stories.unshift(data);
  }
  persist();
  closeSheet(el.addSheet);
  render();
  toast(editingId ? "Cerita diperbarui." : "Cerita berhasil ditambahkan.");
  editingId = null;
}

/* ---------------- Delete (confirm + undo) ---------------- */

function requestDelete() {
  const story = stories.find((s) => s.id === activeStoryId);
  if (!story) return;
  el.alertTitle.textContent = "Hapus Cerita?";
  el.alertMessage.textContent = `“${story.title}” akan dihapus permanen. Tindakan ini bisa diurungkan sebentar setelah dihapus.`;
  el.alertConfirm.textContent = "Hapus";
  showAlert(() => doDelete());
}

function showAlert(onConfirm) {
  el.alertBackdrop.hidden = false;
  requestAnimationFrame(() => el.alertBackdrop.classList.add("is-visible"));
  el.alertCancel.onclick = hideAlert;
  el.alertConfirm.onclick = () => {
    hideAlert();
    onConfirm && onConfirm();
  };
  setTimeout(() => el.alertCancel.focus(), 60);
}

function hideAlert() {
  el.alertBackdrop.classList.remove("is-visible");
  setTimeout(() => {
    if (!el.alertBackdrop.classList.contains("is-visible")) {
      el.alertBackdrop.hidden = true;
    }
  }, 200);
}

function doDelete() {
  const index = stories.findIndex((s) => s.id === activeStoryId);
  if (index === -1) return;
  const [removed] = stories.splice(index, 1);
  deletedSnapshot = { story: removed, index };
  persist();
  closeSheet(el.readSheet);
  render();
  toast("Cerita dihapus.", "Urungkan", undoDelete);
}

function undoDelete() {
  if (!deletedSnapshot) return;
  const { story, index } = deletedSnapshot;
  deletedSnapshot = null;
  if (!stories.some((s) => s.id === story.id)) {
    stories.splice(Math.min(index, stories.length), 0, story);
    persist();
    render();
    toast("Cerita dikembalikan.");
  }
}

/* ---------------- Filter & search ---------------- */

function initFilter() {
  document.querySelectorAll(".segment").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".segment").forEach((b) => {
        b.classList.remove("is-active");
        b.setAttribute("aria-selected", "false");
      });
      btn.classList.add("is-active");
      btn.setAttribute("aria-selected", "true");
      currentFilter = btn.dataset.filter;
      render();
    });
  });
}

function initSearch() {
  el.search.addEventListener("input", () => {
    searchTerm = el.search.value;
    el.searchClear.hidden = el.search.value.length === 0;
    render();
  });
  el.searchClear.addEventListener("click", () => {
    el.search.value = "";
    searchTerm = "";
    el.searchClear.hidden = true;
    el.search.focus();
    render();
  });
}

/* ---------------- Wire up ---------------- */

async function init() {
  initTheme();
  initFilter();
  initSearch();

  // Add buttons
  $("#addButtonTop").addEventListener("click", openAdd);
  $("#emptyAddButton").addEventListener("click", openAdd);

  // Add / edit sheet
  el.cancelButton.addEventListener("click", () => closeSheet(el.addSheet));
  el.saveButton.addEventListener("click", handleSave);
  el.storyForm.addEventListener("submit", (e) => {
    e.preventDefault();
    handleSave();
  });
  el.storyForm.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      handleSave();
    }
  });
  el.contentInput.addEventListener("input", updateCharCount);

  // Read sheet actions
  el.closeReadButton.addEventListener("click", () => closeSheet(el.readSheet));
  el.shareButton.addEventListener("click", shareStory);
  el.favoriteButton.addEventListener("click", () => toggleFavorite(activeStoryId));
  el.editButton.addEventListener("click", () => openEdit(activeStoryId));
  el.deleteButton.addEventListener("click", requestDelete);
  el.readScroll.addEventListener("scroll", updateProgress, { passive: true });

  // Alert
  el.alertCancel.addEventListener("click", hideAlert);

  // Scrim closes whichever sheet is open
  el.scrim.addEventListener("click", () => {
    if (!el.alertBackdrop.hidden) return;
    closeAllSheets();
  });

  // Escape key
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (!el.alertBackdrop.hidden) {
      hideAlert();
      return;
    }
    closeAllSheets();
  });

  await initData();
  render();
}

document.addEventListener("DOMContentLoaded", init);
