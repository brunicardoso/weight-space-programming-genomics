#!/bin/bash
# Operator automation for preregistered runs (not a §9 file; changes no session condition). Run as root:
#
#   sudo bash harness/batch.sh -n <N>            up to N scheduled runs (make_run --next)
#   sudo bash harness/batch.sh --toy             one toy_ca run outside the schedule (archived as a non-result)
#   options: --timeout-min M (no-progress timeout, default 45)  --idle-sec S (idle before /exit, default 60)
#            --env-ref FILE (interactive reference, default runs/batch/interactive_env.txt)
#
# Per run: make_run -> install.sh (stop on refusal/FAIL) -> fp_server in the background -> the session as wspagent in
# a tmux pane (pty owned by wspagent) started with exactly the frozen launch command -> two start-up screens answered
# as the operator answers them interactively, each only when recognised from its full text (batch_helper.py):
#   1. Claude Code's folder-trust question for this run directory: Down to "Yes, I trust this folder", Enter;
#   2. the /mnt/c "Settings Error" screen (preregistration §4.2: answered with option 3): "3". It lists exactly
#      /mnt/c/Program Files/ClaudeCode/managed-settings.d and .../managed-settings.json, each with EACCES, because
#      wspagent cannot read /mnt/c/Program Files; C:\Program Files\ClaudeCode does not exist, so no settings are
#      skipped (operator check, 2026-11-05). Another path, errno or entry does not match.
# Any other selection screen stops the batch with nothing sent. -> wait for phase B accepted (fp_server log) and an
# idle transcript -> "/exit" -> if Claude Code shows its "Background work is running" exit screen (options exactly
# 1. Exit and stop tasks / 2. Move to background and exit / 3. Stay; Amendment 9), "1" (only after phase B was accepted);
# any other selection screen after /exit stops the batch with nothing sent -> tmux server stopped; wspagent processes
# left over from the session (started during the run, on the pane's tty or none) are logged and killed -> finish_run.sh (stop server, archive, audit) -> path_review.py (the audit's
# outside_path findings re-resolved against the shell's working directory; review procedure, summary logged, never
# stops the batch) -> forward_check.py (static check for local forward computation; report in runs/wsp/<id>/, summary
# logged, never stops the batch: Amendment 3 allows local computation and records it as an exploratory measure) ->
# continue on valid / needs-review, and on audit `invalid` when Amendment 4 makes it needs-review (all invalid findings
# are outside_path findings that path_review cleared, none missed). Operational rule of 2026-11-06 (M6 operations log;
# no run's status changes): also continue when every invalid finding is an outside_path finding that path_review marks
# cleared or undetermined and path_review found no outside path the audit missed; the run is logged as "invalid,
# pending review" and listed at the end. Stop on any other audit `invalid` (another category, or an outside_path
# finding path_review confirms, marks not executed / review list, or no usable path_review.json).
# On a stall (no change in the transcript, fp_log or state for --timeout-min), an unexpected screen, or any failure,
# the batch stops and leaves the session, server and run untouched for the operator.
#
# Isolation: the tmux server runs as wspagent; its socket is removed as soon as the session has started (the session
# could otherwise drive tmux through it, outside the Bash sandbox) and recreated (SIGUSR1) only to send /exit once the
# session is idle. The pane's login shell starts from the interactive reference (`env` of a wspagent login in Windows
# Terminal): TERM, WSL_DISTRO_NAME, WSLENV and LANG copied, PATH chosen so that after the login scripts it equals the
# reference PATH, SHLVL so that the login shell has 1, and without the variables tmux adds (TMUX, TMUX_PANE,
# TERM_PROGRAM*, COLORTERM); the window has the reference's size. The only keystrokes are the launch line + Enter,
# Down + Enter on the folder-trust question, "3" on the /mnt/c screen (each if shown), "/exit" + Enter, and "1" on the
# exit screen (if shown). The session's environment is diffed against the reference and logged. Start-up screens are
# logged; after start-up the pane is captured only after /exit, until the session has ended, and only the recognised
# exit screen's frame is logged (title, subtitle, options; not its task lines or anything above it), so no session
# output reaches the logs. meta.json records launch=batch. Every keystroke and
# step goes to runs/batch/.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
AGENT=wspagent
MAIN=mainuser
AGENT_HOME=/home/wspagent
AGENT_RUNS=$AGENT_HOME/runs
FP_SOCK=/run/wsp/fp.sock
TNAME=wsp                                  # tmux -L socket name (under /tmp/tmux-<uid of wspagent>/)
FIXED_PROMPT='Read AGENTS.md and complete the task.'
MODEL=claude-opus-5-5
EFFORT=high
ENV_REF=""; N=1; TOY=0; TIMEOUT_MIN=45; IDLE_SEC=60; POST_MAX_MIN=15; START_MAX=180

usage() { sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
  case "$1" in
    -n) N=$2; shift 2 ;;
    --toy) TOY=1; shift ;;
    --timeout-min) TIMEOUT_MIN=$2; shift 2 ;;
    --idle-sec) IDLE_SEC=$2; shift 2 ;;
    --env-ref) ENV_REF=$2; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "batch: unknown argument $1"; usage; exit 2 ;;
  esac
done
case "$N$TIMEOUT_MIN$IDLE_SEC" in *[!0-9]*|"") echo "batch: -n, --timeout-min and --idle-sec take integers"; exit 2 ;; esac
[ "$TOY" = 1 ] && N=1
[ "$(id -u)" = 0 ] || { echo "batch: run as root: sudo bash $0 $*"; exit 2; }
command -v tmux >/dev/null || { echo "batch: tmux not found"; exit 2; }
PIXI=$(runuser -l "$MAIN" -c 'command -v pixi') || { echo "batch: pixi not found for $MAIN"; exit 2; }

H() { python3 -I "$REPO/harness/batch_helper.py" "$@"; }
T() { runuser -u "$AGENT" -- tmux -L "$TNAME" "$@"; }
claude_running() { pgrep -u "$AGENT" -f '(^|/)claude( |$)' >/dev/null; }
now() { date +%s; }
# wspagent processes on the pane's tty or with no tty, and processes not in the list taken before the session started
agent_procs() { ps -u "$AGENT" -o pid=,tty= 2>/dev/null | awk -v t="${PTY#/dev/}" '$2==t || $2=="?" {print $1}' | sort; }
leftovers() { comm -13 <(printf '%s\n' $PRE | sort) <(agent_procs) | tr '\n' ' ' | sed 's/ *$//'; }

BDIR=$REPO/runs/batch
mkdir -p "$BDIR"; chown "$MAIN:" "$BDIR"
TS=$(date +%Y%m%d-%H%M%S)
LOG=$BDIR/batch_$TS.log
touch "$LOG"; chown "$MAIN:" "$LOG"
PANE=$(mktemp); trap 'rm -f "$PANE"' EXIT
log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
own() { touch "$1"; chown "$MAIN:" "$1"; }

RID=""; RUN=""; TPID=""; TSOCK=""; PENDING=""
pending_list() { [ -z "$PENDING" ] || log "invalid, pending review (decide with the operator after the batch):$PENDING"; }
operator_help() {
  [ -n "$RID" ] || return 0
  log "Left untouched for the operator: run $RID, fp_server ($( [ -S "$FP_SOCK" ] && echo running || echo not running)),"
  log "  session: $(claude_running && echo 'claude running' || echo 'no claude process'), tmux server pid ${TPID:-none}."
  [ -n "$TPID" ] && log "  to see the session: sudo kill -USR1 $TPID; sudo -u $AGENT tmux -L $TNAME attach -t $TNAME"
  log "  after it has exited: sudo sh $REPO/harness/finish_run.sh $RID"
}
stop() { log "STOP: $*"; operator_help; pending_list; log "batch stopped"; exit 3; }
trap 'log "STOP: interrupted by a signal"; operator_help; exit 130' INT TERM HUP
keys_literal() { log "KEYS literal: $1"; T send-keys -t "$TNAME:0.0" -l -- "$1"; }
keys_enter() { log "KEYS Enter"; T send-keys -t "$TNAME:0.0" Enter; }
keys_key() { log "KEYS $1"; T send-keys -t "$TNAME:0.0" "$1"; }
# capture through a pipe (root writes the file): the tmux client runs as wspagent, and a capture redirected into a
# root-owned file came back empty in toy run c1512879eff2741d while the screen was visible
capture() {
  local out err rc
  err=$(mktemp)
  out=$(T capture-pane -p -t "$TNAME:0.0" 2>"$err"); rc=$?
  printf '%s\n' "$out" >"$PANE"
  if [ $rc != 0 ] || [ -z "$(printf '%s' "$out" | tr -d '[:space:]')" ]; then
    log "capture-pane: exit $rc, $(printf '%s' "$out" | wc -c) bytes, stderr: $(tr '\n' ' ' <"$err")"
  fi
  rm -f "$err"
}
snapshot() { log "pane snapshot ($1):"; sed 's/[[:space:]]*$//' "$PANE" | grep -v '^$' | sed 's/^/    | /' | tee -a "$LOG" >/dev/null; }

# interactive reference: TERM, WSL_DISTRO_NAME, WSLENV and the window size of a Windows Terminal login of wspagent
ENV_REF=${ENV_REF:-$BDIR/interactive_env.txt}
[ -f "$ENV_REF" ] || { log "STOP: no interactive environment reference $ENV_REF (capture it first; see the operations log)"; exit 3; }
PANE_ENV="-u TMUX -u TMUX_PANE -u TERM_PROGRAM -u TERM_PROGRAM_VERSION -u COLORTERM"
for v in TERM WSL_DISTRO_NAME WSLENV LANG; do
  val=$(H envref "$ENV_REF" "$v"); rc=$?
  case $rc in
    0) PANE_ENV="$PANE_ENV $v=$val" ;;
    1) PANE_ENV="$PANE_ENV -u $v" ;;
    *) log "STOP: unsafe value for $v in $ENV_REF"; exit 3 ;;
  esac
done
SIZE=$(H envref "$ENV_REF" STTY_SIZE) || { log "STOP: no STTY_SIZE in $ENV_REF"; exit 3; }
ROWS=${SIZE% *}; COLS=${SIZE#* }
case "$ROWS$COLS" in *[!0-9]*|"") log "STOP: bad STTY_SIZE '$SIZE' in $ENV_REF"; exit 3 ;; esac
# PATH: the starting PATH whose login (/etc/profile, ~/.profile) gives exactly the reference PATH, found by dropping
# leading entries of the reference one at a time and simulating an interactive login (.profile and .bashrc) as wspagent
REF_PATH=$(H envref "$ENV_REF" PATH) || { log "STOP: no usable PATH in $ENV_REF"; exit 3; }
PANE_PATH=""; cand=$REF_PATH
for j in 0 1 2 3; do
  got=$(runuser -u "$AGENT" -- env -i HOME="$AGENT_HOME" USER="$AGENT" LOGNAME="$AGENT" SHELL=/bin/bash TERM=xterm-256color \
        PATH="$cand" bash -l -i -c 'echo "$PATH"' 2>/dev/null </dev/null)
  [ "$got" = "$REF_PATH" ] && { PANE_PATH=$cand; break; }
  case $cand in *:*) cand=${cand#*:} ;; *) break ;; esac
done
[ -n "$PANE_PATH" ] || { log "STOP: no starting PATH reproduces the reference PATH after login"; exit 3; }
PANE_ENV="$PANE_ENV PATH=$PANE_PATH SHLVL=0"
PANE_CMD="exec env $PANE_ENV /bin/bash -l"
log "pane: ${COLS}x${ROWS}, login shell: $PANE_CMD (reference $ENV_REF)"

if [ "$TOY" = 1 ]; then
  MRARGS="--target toy_ca --phases AB --budget-a 2 --budget-b 2 --claude-model $MODEL --effort $EFFORT"
else
  MRARGS="--next"
fi
log "batch start: runs=$N toy=$TOY timeout=${TIMEOUT_MIN}min idle=${IDLE_SEC}s repo=$REPO git=$(runuser -u "$MAIN" -- git -C "$REPO" rev-parse --short HEAD)"

done_runs=""
for i in $(seq 1 "$N"); do
  RID=""; RUN=""; TPID=""; TSOCK=""; PTY=""; PRE=""
  log "=== run $i of $N"
  # --- 0. clean slate
  claude_running && stop "a claude process of $AGENT is running"
  pgrep -u "$AGENT" -x tmux >/dev/null || pgrep -u "$AGENT" -f '^tmux' >/dev/null && stop "a tmux server of $AGENT is running"
  [ -z "$(ls -A "$AGENT_RUNS" 2>/dev/null)" ] || stop "$AGENT_RUNS is not empty"
  [ -S "$FP_SOCK" ] && stop "fp_server socket $FP_SOCK exists (server still running?)"

  # --- 1. make_run, install
  MR=$BDIR/${TS}_run${i}_make_run.out; own "$MR"
  log "step make_run: python harness/make_run.py $MRARGS"
  runuser -l "$MAIN" -c "cd '$REPO' && '$PIXI' run python harness/make_run.py $MRARGS" >"$MR" 2>&1
  rc=$?; sed 's/^/    | /' "$MR" >>"$LOG"
  [ $rc = 0 ] || stop "make_run refused (exit $rc)"
  RID=$(H runid "$MR") || stop "could not read the run id from make_run's output"
  RUN=$REPO/runs/wsp/$RID
  log "run $RID prepared"
  m=$(H meta "$RUN" claude_model); e=$(H meta "$RUN" effort); p=$(H meta "$RUN" fixed_prompt)
  [ "$m" = "$MODEL" ] && [ "$e" = "$EFFORT" ] && [ "$p" = "$FIXED_PROMPT" ] \
    || stop "meta.json model/effort/prompt ($m / $e / $p) differ from the frozen launch command"
  runuser -u "$MAIN" -- python3 -I "$REPO/harness/set_launch.py" --run "$RID" --method batch --by batch.sh \
    --detail "batch_log=$(basename "$LOG")" --detail "size=${COLS}x${ROWS}" --detail "env_ref=$(basename "$ENV_REF")" \
    --detail "pane=$PANE_CMD" | tee -a "$LOG" || stop "could not record the launch method in meta.json"
  IN=$BDIR/${TS}_${RID}_install.out; own "$IN"
  log "step install: sh $RUN/install.sh"
  sh "$RUN/install.sh" >"$IN" 2>&1
  rc=$?; sed 's/^/    | /' "$IN" >>"$LOG"
  if [ $rc != 0 ] || grep -q 'FAIL' "$IN" || grep -qi 'refus' "$IN"; then stop "install.sh refused or a check failed (exit $rc)"; fi

  # --- 2. fp_server
  FPLOG=$BDIR/${TS}_${RID}_fp_server.log; own "$FPLOG"
  log "step fp_server: python harness/fp_server.py --run $RID (background, as $MAIN)"
  setsid runuser -l "$MAIN" -c "cd '$REPO' && exec '$PIXI' run python harness/fp_server.py --run $RID" >>"$FPLOG" 2>&1 </dev/null &
  t0=$(now); until [ -S "$FP_SOCK" ]; do [ $(( $(now) - t0 )) -ge 120 ] && stop "fp_server socket did not appear in 120 s"; sleep 1; done
  log "fp_server listening on $FP_SOCK"

  # --- 3. session in tmux as wspagent
  PRE=$(ps -u "$AGENT" -o pid= 2>/dev/null | tr -d ' ')
  log "step tmux: tmux -L $TNAME -f /dev/null new-session -d -s $TNAME -x $COLS -y $ROWS \"$PANE_CMD\" (as $AGENT)"
  runuser -l "$AGENT" -c "tmux -L $TNAME -f /dev/null new-session -d -s $TNAME -x $COLS -y $ROWS \"$PANE_CMD\"" || stop "tmux did not start"
  TPID=$(T display-message -p '#{pid}'); TSOCK=$(T display-message -p '#{socket_path}')
  PTY=$(T display-message -p -t "$TNAME:0.0" '#{pane_tty}')
  log "tmux server pid $TPID, socket $TSOCK, pane tty $PTY owned by $(stat -c %U "$PTY")"
  [ "$(stat -c %U "$PTY")" = "$AGENT" ] || stop "pane tty is not owned by $AGENT"
  sleep 2
  LAUNCH="cd ~/runs/$RID && claude --model $MODEL --effort $EFFORT \"$FIXED_PROMPT\""
  keys_literal "$LAUNCH"; keys_enter

  # --- 4. start-up: the folder-trust question (Down to "Yes, I trust this folder", Enter) and the /mnt/c "Settings
  #        Error" screen for the two EACCES managed-settings paths ("3", preregistration §4.2), each recognised from
  #        its full text; any other selection screen stops the batch with nothing sent.
  #        Every new screen state and the screen before each keystroke are logged (start-up only).
  TDIR=$AGENT_HOME/.claude/projects/-home-$AGENT-runs-$RID
  t0=$(now); sent3=""; down=""; trusted=""; started=""; prev=""
  while [ $(( $(now) - t0 )) -lt $START_MAX ]; do
    if ls "$TDIR"/*.jsonl >/dev/null 2>&1; then started=1; break; fi
    capture
    c=$(H screen "$PANE" "$AGENT_RUNS/$RID")
    sum=$(md5sum <"$PANE" | cut -c1-12)
    [ "$c:$sum" != "$prev" ] && { snapshot "screen: $c"; prev="$c:$sum"; }
    case $c in
      trust_no)
        if [ -z "$down" ]; then keys_key Down; down=$(now)
        elif [ $(( $(now) - down )) -ge 10 ]; then stop "trust question: cursor still on \"No, exit\" 10 s after Down; nothing more sent"; fi ;;
      trust_yes)
        if [ -z "$trusted" ]; then keys_enter; trusted=$(now)
        elif [ $(( $(now) - trusted )) -ge 15 ]; then stop "trust question still shown 15 s after Enter"; fi ;;
      mnt_c)
        [ -n "$trusted" ] || log "note: /mnt/c screen shown before any trust question"
        if [ -z "$sent3" ]; then keys_literal "3"; sent3=$(now)
        elif [ $(( $(now) - sent3 )) -ge 15 ]; then stop "/mnt/c screen still shown 15 s after \"3\""; fi ;;
      menu) stop "unrecognised selection screen at start-up; nothing sent" ;;
    esac
    sleep 2
  done
  if [ -z "$started" ]; then capture; snapshot "no transcript yet"; stop "the session did not start within ${START_MAX} s"; fi
  [ -n "$trusted" ] || log "no folder-trust question was shown"
  [ -n "$sent3" ] || log "no /mnt/c settings screen was shown"
  rm -f "$TSOCK"
  T list-sessions >/dev/null 2>&1 && stop "tmux socket still reachable after removal"
  log "session started (transcript in $TDIR); tmux socket removed until /exit"
  CPID=$(pgrep -o -u "$AGENT" -f '(^|/)claude( |$)')
  if [ -n "$CPID" ]; then
    CE=$BDIR/${TS}_${RID}_claude_env.txt; own "$CE"
    tr '\0' '\n' <"/proc/$CPID/environ" >"$CE"
    log "session environment vs interactive reference (claude pid $CPID, parent $(ps -o comm= -p "$(ps -o ppid= -p "$CPID" | tr -d ' ')")):"
    H envdiff "$ENV_REF" "$CE" | sed 's/^/    ENV /' | tee -a "$LOG"
  else
    log "WARNING: no claude process found to read its environment"
  fi

  # --- 5. wait for phase B accepted; stop on a stall
  sig=""; last=$(now); beat=$(now); prev=""
  while :; do
    sleep 30
    st=$(H state "$RUN"); s=$(H progress "$TDIR" "$RUN/fp_log.jsonl" "$RUN/state.json")
    [ "$s" != "$sig" ] && { sig=$s; last=$(now); }
    [ "$st" != "$prev" ] && { log "server state: $st"; prev=$st; }
    case $st in *b_accepted=1*) log "phase B accepted"; break ;; esac
    claude_running || stop "claude exited before phase B was accepted"
    [ -S "$FP_SOCK" ] || stop "fp_server socket disappeared"
    [ $(( $(now) - last )) -ge $(( TIMEOUT_MIN * 60 )) ] && stop "no progress for ${TIMEOUT_MIN} min (transcript, fp_log, state unchanged)"
    [ $(( $(now) - beat )) -ge 600 ] && { log "monitor: $st, last change $(( $(now) - last )) s ago"; beat=$(now); }
  done

  # --- 6. idle transcript, then /exit
  t0=$(now); sig=$(H progress "$TDIR"); last=$(now)
  while claude_running; do
    sleep 10
    s=$(H progress "$TDIR"); [ "$s" != "$sig" ] && { sig=$s; last=$(now); }
    [ $(( $(now) - last )) -ge "$IDLE_SEC" ] && break
    [ $(( $(now) - t0 )) -ge $(( POST_MAX_MIN * 60 )) ] && stop "session still active ${POST_MAX_MIN} min after phase B"
  done
  if claude_running; then
    log "transcript idle for ${IDLE_SEC} s; recreating the tmux socket (SIGUSR1 to $TPID)"
    kill -USR1 "$TPID"
    t1=$(now); until T list-sessions >/dev/null 2>&1; do [ $(( $(now) - t1 )) -ge 10 ] && stop "tmux socket not recreated"; sleep 1; done
    keys_literal "/exit"; sleep 1; keys_enter
    # Amendment 9: Claude Code's exit screen for running background work is answered "1" (Exit and stop tasks), only
    # when recognised exactly and only after phase B was accepted; its frame is logged, never its task lines or the
    # screen above it; any other selection screen stops the batch with nothing sent (and nothing logged)
    t1=$(now); sent1=""
    while claude_running; do
      [ $(( $(now) - t1 )) -ge 90 ] && { : >"$PANE"; stop "/exit did not end the session within 90 s"; }
      sleep 2
      claude_running || break
      capture
      c=$(H exitscreen "$PANE")
      case $c in
        exit_tasks)
          if [ -z "$sent1" ]; then
            case $(H state "$RUN") in *b_accepted=1*) ;; *) : >"$PANE"; stop "exit screen shown but phase B not accepted; nothing sent" ;; esac
            log "exit screen after /exit (background work running; task lines and the screen above not logged):"
            H exitsnap "$PANE" | sed 's/^/    | /' | tee -a "$LOG" >/dev/null
            keys_literal "1"; sent1=$(now)
          elif [ $(( $(now) - sent1 )) -ge 15 ]; then : >"$PANE"; stop "exit screen still shown 15 s after \"1\""; fi ;;
        menu) : >"$PANE"; stop "unrecognised selection screen after /exit; nothing sent (not logged: it can show session output)" ;;
      esac
    done
    : >"$PANE"
    log "session exited${sent1:+ after \"1\" (Exit and stop tasks)}"
  else
    log "session exited on its own after phase B"
  fi
  T kill-server 2>/dev/null || kill "$TPID" 2>/dev/null
  log "tmux server stopped"; TPID=""
  # no wspagent process from the session may outlive it (e.g. background tasks): log (pid, ppid, age, tty, name) and kill
  sleep 2; left=$(leftovers)
  if [ -n "$left" ]; then
    log "leftover $AGENT process(es) from the session (pid ppid elapsed-s tty name):"
    ps -o pid=,ppid=,etimes=,tty=,comm= -p "$(echo $left | tr ' ' ,)" | sed 's/^/    | /' | tee -a "$LOG" >/dev/null
    kill -TERM $left 2>/dev/null; sleep 5
    l2=$(leftovers); [ -n "$l2" ] && { log "still running after SIGTERM: $l2; SIGKILL"; kill -KILL $l2 2>/dev/null; sleep 1; }
    l3=$(leftovers); [ -n "$l3" ] && stop "$AGENT process(es) $l3 survived SIGKILL"
    log "leftover processes stopped: $left"
  else
    log "no $AGENT process from the session remains"
  fi

  # --- 7. finish (stop server, archive, audit)
  FO=$BDIR/${TS}_${RID}_finish.out; own "$FO"
  log "step finish: sh harness/finish_run.sh $RID (output in $FO)"
  sh "$REPO/harness/finish_run.sh" "$RID" >"$FO" 2>&1
  rc=$?
  [ -f "$RUN/audit.json" ] || stop "finish_run.sh failed (exit $rc), no audit.json"
  grep -q '^WARNING' "$FO" && log "finish warnings: $(grep '^WARNING' "$FO" | tr '\n' ' ')"
  a=$(H audit "$RUN"); log "AUDIT $RID: $a"
  done_runs="$done_runs $RID:${a%% *}"

  # --- 8. review-procedure checks (not the frozen audit), logged only, also for an invalid run; reports in runs/wsp/<id>/
  #   path_review.py: the audit's outside_path findings re-resolved against the shell's working directory
  #   forward_check.py: local forward computation (Amendment 3: exploratory measure)
  pr=$(runuser -l "$MAIN" -c "cd '$REPO' && '$PIXI' run python harness/path_review.py --run $RID --summary" 2>&1 | tail -n 1)
  case $pr in
    "path_review $RID: "*) log "$pr" ;;
    *) log "WARNING: path_review.py failed for $RID (rerun: pixi run python harness/path_review.py --run $RID): $pr" ;;
  esac
  fc=$(runuser -l "$MAIN" -c "cd '$REPO' && '$PIXI' run python harness/forward_check.py --run $RID \
        --md runs/wsp/$RID/forward_check.md --json runs/wsp/$RID/forward_check.json --summary" 2>&1 | tail -n 1)
  case $fc in      # logged only (Amendment 3); a failed check is logged and can be rerun later
    "forward_check $RID: "*) log "$fc" ;;
    *) log "WARNING: forward_check.py failed for $RID (rerun: pixi run python harness/forward_check.py --run $RID --json runs/wsp/$RID/forward_check.json --md runs/wsp/$RID/forward_check.md): $fc" ;;
  esac
  case $a in invalid*)
    bs=$(H batchstatus "$RUN" 2>&1 | tail -n 1)
    case $bs in
      "continue needs-review (A4)")      # Amendment 4: every invalid finding is a cleared outside_path finding
        log "Amendment 4: $RID audit invalid, but all invalid findings are outside_path findings cleared by path_review (none missed): needs-review, decided later from the packet; continuing"
        done_runs="$done_runs(A4:needs-review)" ;;
      "pending "*)                      # operational rule 2026-11-06: only cleared / undetermined outside_path findings
        log "invalid, pending review: $RID audit invalid, ${bs#pending }; no other invalid finding; continuing (status unchanged, decided after the batch)"
        done_runs="$done_runs(invalid,pending-review)"; PENDING="$PENDING $RID" ;;
      *)
        RID=""; stop "audit invalid (${bs#stop }); batch stopped for the operator (path_review above; packet: harness/review_packet.py)" ;;
    esac ;;
  esac
  RID=""
done
log "batch done:$done_runs"
pending_list
