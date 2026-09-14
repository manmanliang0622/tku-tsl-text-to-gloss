#!/usr/bin/env bash
# 把主機的動作庫索引發佈到前端——補片入庫的最後一步（2026-09-14 補上）。
#
# 換片要三份 lexicon 都更新，前兩份入庫流程已經處理：
#   主機 ~/0813/recordings/lexicon.json          ingest_new_videos.py --apply 寫的
#   主機 ~/0821_bundle/recordings/lexicon.json   rsync 過去的，後端 /clips 讀這份
#   Pages repo 的 recordings/lexicon.json        ← 這支負責的那份
#
# 少了第三份會怎樣：前端 composer.loadLexicon() 抓的是**相對路徑**
# recordings/lexicon.json，在 pages.dev 上那是 Pages 自己帶的靜態檔，不是主機那份
# （只有 /translate 與 /clips 走 signavatar-config.js 的 /api 代理）。索引停在舊版，
# 前端就照著舊索引去跟主機要**舊片**——畫面上像「影片根本沒換」，但查主機兩份都是對的，
# 從主機那端完全看不出問題。2026-08-21 到 09-14 就是這樣：七批補片、716 個詞改指、
# 43 個新鍵，全部沒在網站上生效。
#
# 用法（在本機跑，需要 gh 已登入）：
#     scripts/publish_lexicon.sh              # 檢查 → 發佈 → 等部署生效
#     scripts/publish_lexicon.sh --dry-run    # 只比對，不推
#
# 環境變數：
#     TSL_VM_HOST    主機（預設 tku-gpu，見 HANDOFF §2；本 repo 公開，位址不寫進來）
#     TSL_VM_PORT    非 22 的埠；用 ~/.ssh/config 的 alias 時不必設
#     TSL_PAGES_REPO 前端 repo（預設 manmanliang0622/TKU-TSL-Avatar-Project）
#     TSL_PAGES_URL  對外網址（預設 https://signbridge-2co.pages.dev）
set -u

HOST="${TSL_VM_HOST:-tku-gpu}"
PORT="${TSL_VM_PORT:-}"
# ssh 用 -p、scp 用 -P，同一個埠兩種旗標。展開寫成 ${A[@]+…} 是因為
# macOS 內建 bash 3.2 在 set -u 下展開空陣列會直接報 unbound variable。
SSH_P=(); SCP_P=()
[ -n "$PORT" ] && { SSH_P=(-p "$PORT"); SCP_P=(-P "$PORT"); }
REPO="${TSL_PAGES_REPO:-manmanliang0622/TKU-TSL-Avatar-Project}"
SITE="${TSL_PAGES_URL:-https://signbridge-2co.pages.dev}"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

count() {  # 數 lexicon 條目；讀不到就印 0，讓呼叫端自己判斷
  python3 -c 'import json,sys
try: print(len(json.load(open(sys.argv[1], encoding="utf-8"))))
except Exception: print(0)' "$1"
}

# ── 1) 抓主機那份，並確認後端真的播得出來 ────────────────────────────
# 先問 0821_bundle：lexicon 指到的錄影檔有沒有缺。rsync 的 --include 是寫死前綴的，
# 換批次前綴忘了改 filter，片就不會過去；那個詞在前端會變成 /clips 回 missing。
echo "== 檢查主機 ($HOST) =="
if ! ssh ${SSH_P[@]+"${SSH_P[@]}"} -o BatchMode=yes -o ConnectTimeout=15 "$HOST" 'test -f ~/0813/recordings/lexicon.json' 2>/dev/null; then
  echo "連不到 $HOST，或主機上沒有 ~/0813/recordings/lexicon.json。" >&2
  echo "設 TSL_VM_HOST=<user@host>，或在 ~/.ssh/config 加一個 tku-gpu 的 Host。" >&2
  exit 1
fi

MISSING="$(ssh ${SSH_P[@]+"${SSH_P[@]}"} -o BatchMode=yes "$HOST" 'cd ~/0821_bundle/recordings && python3 -c "
import json, os
lex = json.load(open(\"lexicon.json\", encoding=\"utf-8\"))
have = set(os.listdir(\".\"))
bad = sorted({v[\"recording\"] for v in lex.values()
              if v.get(\"recording\") and v[\"recording\"] not in have})
print(len(bad))
for r in bad[:10]: print(\"  \", r)
"')"
if [ "$(printf '%s' "$MISSING" | head -1)" != "0" ]; then
  echo "0821_bundle 缺錄影檔，前端會拿到 missing——先補 rsync 再發佈：" >&2
  printf '%s\n' "$MISSING" >&2
  exit 1
fi
echo "  0821_bundle 錄影檔齊全"

# 主機兩份要一致，否則 /clips（讀 0821_bundle）跟前端索引會對不起來
H813="$(ssh ${SSH_P[@]+"${SSH_P[@]}"} -o BatchMode=yes "$HOST" 'md5sum ~/0813/recordings/lexicon.json | cut -d" " -f1')"
H821="$(ssh ${SSH_P[@]+"${SSH_P[@]}"} -o BatchMode=yes "$HOST" 'md5sum ~/0821_bundle/recordings/lexicon.json | cut -d" " -f1')"
if [ "$H813" != "$H821" ]; then
  echo "主機兩份 lexicon 不一致（0813 $H813 / 0821_bundle $H821），先 rsync。" >&2
  exit 1
fi
echo "  0813 與 0821_bundle 一致 ($H813)"

scp ${SCP_P[@]+"${SCP_P[@]}"} -q "$HOST:~/0813/recordings/lexicon.json" "$WORK/new.json" || exit 1
NEW="$(count "$WORK/new.json")"
[ "$NEW" -gt 0 ] || { echo "抓下來的 lexicon 不是合法 JSON" >&2; exit 1; }

# ── 2) 跟現在線上那份比 ──────────────────────────────────────────────
echo "== 比對線上 ($SITE) =="
curl -sS --max-time 120 --compressed "$SITE/recordings/lexicon.json" -o "$WORK/live.json" || exit 1
LIVE="$(count "$WORK/live.json")"
python3 - "$WORK/live.json" "$WORK/new.json" "$WORK/diff.json" <<'PY'
import json, sys
live = json.load(open(sys.argv[1], encoding="utf-8"))
new = json.load(open(sys.argv[2], encoding="utf-8"))
added = sorted(set(new) - set(live))
removed = sorted(set(live) - set(new))
changed = sorted(k for k in set(new) & set(live) if new[k] != live[k])
print(f"  線上 {len(live)} 條 / 主機 {len(new)} 條")
print(f"  新增鍵 {len(added)}、改指 {len(changed)}、消失 {len(removed)}")
for label, keys in (("新增", added), ("改指", changed), ("消失", removed)):
    if keys:
        print(f"    {label}: {'、'.join(keys[:12])}{' …' if len(keys) > 12 else ''}")
# 第 4 步驗收用：部署生效＝這些鍵在線上都變成主機的版本
json.dump({"set": added + changed, "gone": removed},
          open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False)
PY

if cmp -s "$WORK/live.json" "$WORK/new.json"; then
  echo "線上已是最新，不用發佈。"
  exit 0
fi
[ "$DRY" -eq 1 ] && { echo "--dry-run：到此為止。"; exit 0; }

# ── 3) 推上 Pages repo ───────────────────────────────────────────────
# .gitignore 有 recordings/，但 recordings/lexicon.json 是從 GitHub 網頁版上傳進去的、
# 已經被追蹤，所以 git add 要 -f，否則會被 ignore 規則擋下來。
NAME="$(git config --global user.name || true)"
EMAIL="$(git config --global user.email || true)"
if [ -z "$NAME" ] || [ -z "$EMAIL" ]; then
  echo "git 全域 user.name／user.email 沒設，commit 會掛成主機名那種來路不明的作者。" >&2
  echo "先設好再跑：git config --global user.name ... / user.email ..." >&2
  exit 1
fi

echo "== 發佈到 $REPO =="
gh repo clone "$REPO" "$WORK/repo" -- --depth 1 --quiet || exit 1
cp "$WORK/new.json" "$WORK/repo/recordings/lexicon.json"
git -C "$WORK/repo" add -f recordings/lexicon.json || exit 1
git -C "$WORK/repo" commit -q -m "詞庫同步到主機現況：$NEW 條（$LIVE → $NEW）

補片入庫後前端的靜態索引要跟著換，否則前端照舊索引去要舊片。
由 scripts/publish_lexicon.sh 產生。" || exit 1
git -C "$WORK/repo" push -q origin HEAD || exit 1
echo "  已推送 $(git -C "$WORK/repo" rev-parse --short HEAD)"

# ── 4) 等 Cloudflare Pages 重新部署（通常 30 秒內） ───────────────────
# 驗收看的是第 2 步比出來的那些鍵，不是條目數：純改指（沒加鍵）時舊版也是同一個
# 條目數，只比數字會在部署前第 1 次輪詢就誤報生效——2026-09-14 換「早安」就是這樣，
# 實際又過了約 40 秒才換。?t= 是繞過快取，免得拿到舊的回應。
# 驗收程式先寫成檔案：bash 3.2 解析 $( ) 裡的 heredoc 會被括號與引號絆倒。
cat > "$WORK/verify.py" <<'PY'
import json, sys
new = json.load(open(sys.argv[2], encoding="utf-8"))
diff = json.load(open(sys.argv[3], encoding="utf-8"))
total = len(diff["set"]) + len(diff["gone"])
try:
    live = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:           # 沒抓到或抓到一半：當作全部還沒換
    print(total, total, "（線上那份讀不到）"); sys.exit()
stale = ([k for k in diff["set"] if live.get(k) != new[k]]
         + [k for k in diff["gone"] if k in live])
print(len(stale), total, "、".join(stale[:5]) + (" …" if len(stale) > 5 else ""))
PY

echo "== 等部署生效 =="
for i in $(seq 1 40); do
  curl -sS --max-time 60 --compressed "$SITE/recordings/lexicon.json?t=$(date +%s)" \
    -o "$WORK/chk.json" 2>/dev/null
  OUT="$(python3 "$WORK/verify.py" "$WORK/chk.json" "$WORK/new.json" "$WORK/diff.json")"
  STALE="$(printf '%s' "$OUT" | cut -d' ' -f1)"
  TOTAL="$(printf '%s' "$OUT" | cut -d' ' -f2)"
  LIST="$(printf '%s' "$OUT" | cut -d' ' -f3-)"
  # 鍵都沒變、只是格式不同（cmp 判不同但比不出鍵）：退回逐位元比對
  if [ "$TOTAL" = "0" ]; then
    cmp -s "$WORK/chk.json" "$WORK/new.json" && STALE=0 || STALE=1
  fi
  if [ "$STALE" = "0" ]; then
    echo "  這次改到的 $TOTAL 個鍵線上都已是新版（第 $i 次輪詢）。網頁重新整理就會換。"
    exit 0
  fi
  echo "  第 $i 次：還有 $STALE/$TOTAL 個鍵是舊的 $LIST"
  sleep 15
done
echo "輪詢 10 分鐘仍未更新，去 Cloudflare Pages 看這次部署的狀態。" >&2
exit 1
