from pathlib import Path
import hashlib,json,os,statistics,struct,subprocess,time,zlib
root=Path.cwd();base=root/'.workspaces/zig-image-decode-20261022';out=base/'real';out.mkdir();fixtures=out/'fixtures';fixtures.mkdir()
def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data))
def png(width,height,kind,rows):return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,kind,0,0,0))+chunk(b'IDAT',zlib.compress(b''.join(b'\0'+r for r in rows)))+chunk(b'IEND',b'')
rgb=bytes([255,0,0,0,255,0,0,0,255,255,255,0,255,0,255,255,255,255]);gray=bytes([0,64,128,192,255,7])
for name,kind,channels,pixels in [('rgb',2,3,rgb),('gray',0,1,gray),('rgba',6,4,b''.join(rgb[i:i+3]+bytes([i*7%256]) for i in range(0,len(rgb),3)))]:
 (fixtures/(name+'.png')).write_bytes(png(2,3,kind,[pixels[i:i+2*channels] for i in range(0,len(pixels),2*channels)]))
source=root/'tests/vision-fixtures/jpeg/prog_ac_refine_zrl_gray.jpg';encoded=source.read_bytes()
for endian,prefix in [('<',b'II'),('>',b'MM')]:
 for orientation in range(1,9):
  tiff=prefix+struct.pack(endian+'HIH',42,8,1)+struct.pack(endian+'HHI',0x112,3,1)+struct.pack(endian+'H',orientation)+b'\0\0'+struct.pack(endian+'I',0)
  payload=b'Exif\0\0'+tiff;(fixtures/f'exif-{prefix.decode()}-{orientation}.jpg').write_bytes(encoded[:2]+b'\xff\xe1'+struct.pack('>H',len(payload)+2)+payload+encoded[2:])
# Warm decode includes the real codec, RGB ownership/orientation and SHA identity.
large=fixtures/'512-rgb.png';large.write_bytes(png(512,512,2,[bytes([y%256,71,193])*512 for y in range(512)]))
paths=[source,root/'tests/vision-fixtures/jpeg/prog_ac_refine_zrl_420.jpg',*sorted(fixtures.glob('*'))];rows=[];references=None
for arm in ['c','native','c-debug','native-debug']:
 binary=base/arm/'bin/test-image-decode';cmd=[str(binary),*map(str,paths)];start=time.monotonic();p=subprocess.run(cmd,capture_output=True,timeout=120)
 (out/(arm+'.stdout')).write_bytes(p.stdout);(out/(arm+'.stderr')).write_bytes(p.stderr);assert p.returncode==0,arm
 records=[json.loads(line) for line in p.stdout.splitlines()]
 if references is None:references=records
 assert records==references,arm+' pixel/fingerprint mismatch'
 for record in records:
  assert record['fingerprint']==hashlib.sha256(struct.pack('=II',record['width'],record['height'])+bytes.fromhex(record['rgb'])).hexdigest()
 row=dict(label=arm,command=cmd,seconds=time.monotonic()-start,returncode=p.returncode,identical_pixels=True,cases=len(records),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest());rows.append(row)
 (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({k:row[k] for k in ['label','seconds','cases']}),flush=True)
by_path=dict(zip(paths,references));assert bytes.fromhex(by_path[fixtures/'rgb.png']['rgb'])==rgb;assert bytes.fromhex(by_path[fixtures/'rgba.png']['rgb'])==rgb;assert bytes.fromhex(by_path[fixtures/'gray.png']['rgb'])==b''.join(bytes([x])*3 for x in gray)
record=by_path[source];pixels=bytes.fromhex(record['rgb']);width=record['width'];matrix=[[pixels[(y*width+x)*3:(y*width+x+1)*3] for x in range(width)] for y in range(record['height'])]
transposed=list(map(list,zip(*matrix)))
expected={1:matrix,2:[r[::-1] for r in matrix],3:[r[::-1] for r in matrix[::-1]],4:matrix[::-1],5:transposed,6:[r[::-1] for r in transposed],7:[r[::-1] for r in transposed[::-1]],8:transposed[::-1]}
for prefix in ['II','MM']:
 for orientation,want in expected.items():
  r=by_path[fixtures/f'exif-{prefix}-{orientation}.jpg'];assert (r['width'],r['height'])==(len(want[0]),len(want));assert bytes.fromhex(r['rgb'])==b''.join(p for line in want for p in line)
for i,arm in enumerate(['c','native','native','c']):
 binary=base/arm/'bin/test-image-decode';cmd=[str(binary),'--bench','64',str(large)];p=subprocess.run(cmd,capture_output=True,timeout=120);assert p.returncode==0
 metric=json.loads(p.stdout);row=dict(label=f'{i}-{arm}-bench',command=cmd,returncode=p.returncode,metrics=metric,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest());rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
