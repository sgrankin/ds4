from pathlib import Path
import hashlib,json,os,re,struct,subprocess,time,zlib
root=Path.cwd();base=root/'.workspaces/zig-combined-20261022';out=base/'vision';out.mkdir()
width,height=112,84
def chunk(name,data):return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data))
png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+bytes([255,0,0])*width)*height))+chunk(b'IEND',b'')
image=out/'red.png';image.write_bytes(png)
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))};env.pop('LINENOISE_ASSUME_TTY',None)
profile='(version 1)(allow default)(deny file-write* (literal '+json.dumps(str(Path.home()/'.ds4_history'))+'))'
rows=[];reference=None
for arm in ['c','native']:
 binary=base/arm/'bin/ds4'
 command=['/usr/bin/sandbox-exec','-p',profile,str(binary),'--model',str(root/'gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf'),'--vision',str(root/'gguf/DeepSeek-V4.1-Flash-1017-vision-BF16.gguf'),'--ssd-streaming','--ssd-streaming-cache-experts','48GB','--ctx','4096','--prefill-chunk','512','--mtp-draft','1','--seed','1','--temp','0','--nothink','--tokens','16','--system','When shown an image, identify its main color. Reply with exactly one uppercase color word and nothing else.']
 stdin=f'/nothink\n/read {image}\nWhat was the main color? Reply only the uppercase color word.\n/exit\n'.encode();(out/(arm+'.stdin')).write_bytes(stdin)
 start=time.monotonic();p=subprocess.run(command,input=stdin,env=env,capture_output=True,timeout=300)
 (out/(arm+'.stdout')).write_bytes(p.stdout);(out/(arm+'.stderr')).write_bytes(p.stderr)
 if reference is None:reference=p.stdout
 row=dict(arm=arm,command=command,returncode=p.returncode,seconds=time.monotonic()-start,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),identical_output=p.stdout==reference,red_answers=len(re.findall(rb'\bRED\b',p.stdout)),image_encoded=bool(re.search(rb'image 112x84, \d+ image tokens',p.stderr)))
 rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
 assert p.returncode==0 and row['image_encoded'] and row['identical_output'] and row['red_answers']>=2,arm
