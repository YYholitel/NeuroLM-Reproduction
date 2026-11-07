"""Source-only ephemeral envelope key; plaintext upload URLs never leave servers."""
import base64,json,os,sys
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import rsa,padding
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
root=Path('/data/lihy/NeuroLM/outputs/vq-five-full-20261107-r1')
private=root/'transport_envelope_private.pem'
if sys.argv[1]=='init':
    if not private.exists():
        key=rsa.generate_private_key(public_exponent=65537,key_size=3072)
        fd=os.open(private,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        with os.fdopen(fd,'wb') as f:
            f.write(key.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
    key=serialization.load_pem_private_key(private.read_bytes(),password=None)
    sys.stdout.buffer.write(key.public_key().public_bytes(serialization.Encoding.PEM,serialization.PublicFormat.SubjectPublicKeyInfo))
elif sys.argv[1]=='decrypt':
    envelope=json.load(sys.stdin)
    key=serialization.load_pem_private_key(private.read_bytes(),password=None)
    secret=key.decrypt(base64.b64decode(envelope['key']),padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),algorithm=hashes.SHA256(),label=None))
    plaintext=AESGCM(secret).decrypt(base64.b64decode(envelope['nonce']),base64.b64decode(envelope['ciphertext']),b'vq-five-full-20261107-r1')
    sys.stdout.buffer.write(plaintext)
