"""Read trusted project pickle labels without reading large EEG byte buffers.

Only binary byte-array payloads over 64 KiB are replaced by equal-length zero
bytes in memory. Opcodes, memo references and small scalar/label buffers remain
unchanged. This is NOT a waveform or complete-file-integrity audit. Input files
are opened read-only. Validate this reader against normal pickle loads first.
"""
import io
import pickle
import pickletools
import struct

OPS={ord(op.code):op for op in pickletools.opcodes}
LARGE={ord('B'):('<I',4),ord('\x8e'):('<Q',8),ord('\x96'):('<Q',8)}


def read_label_envelope(path):
    result=io.BytesIO();skipped=0
    with open(path,'rb') as stream:
        while True:
            code=stream.read(1)
            if not code:raise ValueError('Pickle missing STOP')
            number=code[0]
            if number==0x95:  # FRAME is an optional framing hint.
                if len(stream.read(8))!=8:raise ValueError('Truncated frame')
                continue
            result.write(code)
            if number in LARGE:
                fmt,size=LARGE[number];raw_length=stream.read(size)
                length=struct.unpack(fmt,raw_length)[0];result.write(raw_length)
                if length>65536:
                    stream.seek(length,1);result.write(bytes(length));skipped+=length
                else:
                    data=stream.read(length)
                    if len(data)!=length:raise ValueError('Truncated binary payload')
                    result.write(data)
            else:
                op=OPS[number]
                if op.arg is not None:
                    start=stream.tell();op.arg.reader(stream);end=stream.tell()
                    stream.seek(start);result.write(stream.read(end-start))
            if number==ord('.'):
                if stream.read(1):raise ValueError('Unexpected trailing pickle bytes')
                break
    return result.getvalue(),skipped


def load_labels(path):
    envelope,skipped=read_label_envelope(path)
    return pickle.loads(envelope),skipped


if __name__=='__main__':
    import tempfile
    from pathlib import Path
    import numpy as np
    cases=0
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'example.pkl'
        for protocol in (2,3,4,5):
            for length in (10,20000,100000):
                for label in (0,1,6,np.array([2.]),np.array([4],dtype='int64')):
                    original={'X':np.arange(length,dtype='float64').reshape(1,-1),'ch_names':['FZ'],'y':label}
                    path.write_bytes(pickle.dumps(original,protocol=protocol))
                    actual,skipped=load_labels(path)
                    assert np.array_equal(actual['y'],label) and actual['ch_names']==['FZ']
                    assert actual['X'].shape==original['X'].shape
                    if skipped:assert not actual['X'].any()
                    cases+=1
    print('reader_roundtrip_cases_passed',cases)
