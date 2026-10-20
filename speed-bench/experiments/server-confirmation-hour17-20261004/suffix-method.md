# Fresh-facts suffix confirmation

Run `suffix_sweep.py` with ROOT's Framework Python inside one pressure guard.
It runs reuse64 first, then native; each arm has a new disk cache and response
store, two server lifetimes, and the same absolute advertised file paths.

The workload adapts the hour12 workflow: two sessions read actual active JSON,
then read an archive after an ordinary marked-span deletion. A third request
causes slot turnover. After restart, each archive gets a new label, the second
marked span is deleted, and both sessions must read the changed file again.
There are seven graded steps and six required real reads per arm.

Independent totals are A=`23*14-9=313` and B=`31*8-17=231`. Labels are
`BEACON-46`→`BEACON-47` and `ORBIT-57`→`ORBIT-58`. These are fresh facts/log
wording in an already used workflow family. The packet has not been run.

Cold source warmup, ordinary edited latency, changed-file restart stress,
turnover, startup/restart and whole launcher time remain separate. Task pass
requires the actual fresh read, correct fields and completed response; it
does not require graft activation. Odd alignment, fallback, errors and caps
remain visible. Server log lines retain actual reused/fresh counts and
approximate SSD source flags. Generated histories may differ, so there is no
paired NLL claim.

Only suffix reuse changes between arms. Resident GAP is zero, reasoning is
none, temperature zero, context8192, expert cache64GB, chunk512, selected64/1024,
two slots, SSD hot tier zero. The sent Responses seed is ignored. The sweep
has a600s campaign timeout including both arms; timed-out work is terminated
and retained rather than retried. ROOT alone launches it.
