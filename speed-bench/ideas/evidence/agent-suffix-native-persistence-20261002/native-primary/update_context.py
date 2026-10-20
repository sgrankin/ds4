from pathlib import Path
import os
p=Path('/Volumes/Models/ds4/speed-bench/ideas/artifacts/suffix-agent-persistence-primary/context.txt')
before=p.read_text()
old='Record Otter current access code is 2718.'
new='Record Otter current access code is 3141.'
assert before.count(old)==1
after=before.replace(old,new,1)
(p.parent/'before-tool-edit.txt').write_text(before)
(p.parent/'after-tool-edit.txt').write_text(after)
tmp=p.with_suffix('.edited')
tmp.write_text(after)
os.replace(tmp,p)
print('Updated exactly one earlier record; all other context bytes preserved.')
