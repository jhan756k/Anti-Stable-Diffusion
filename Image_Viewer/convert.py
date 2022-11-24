import os, cv2, zlib
from Crypto.PublicKey import RSA
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
from Crypto.Cipher import AES, PKCS1_OAEP

path = os.getcwd()
img = cv2.imread("{p}/images/original_image/forest.png".format(p=path))

text = str(img.shape[0]).zfill(4) + str(img.shape[1]).zfill(4) + str(img.shape[2]).zfill(4)

for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        text += (str(img[x, y][0]).zfill(3) + str(img[x, y][1]).zfill(3) + str(img[x, y][2]).zfill(3))

key = RSA.generate(2048)
private_key = key.export_key()
file_out = open("private.pem", "wb")
file_out.write(private_key)
file_out.close()

public_key = key.publickey().export_key()
file_out = open("receiver.pem", "wb")
file_out.write(public_key)
file_out.close()

data = text.encode("utf-8")
file_out = open("forest.jhp", "wb")
recipient_key = RSA.import_key(open("receiver.pem").read())
session_key = get_random_bytes(16)
cipher_rsa = PKCS1_OAEP.new(recipient_key)
enc_session_key = cipher_rsa.encrypt(session_key)
cipher_aes = AES.new(session_key, AES.MODE_EAX)
ciphertext, tag = cipher_aes.encrypt_and_digest(data)

# comp_text = zlib.compress(text.encode("utf-8"), 9)
[ file_out.write(x) for x in (enc_session_key, cipher_aes.nonce, tag, ciphertext) ]
file_out.close()