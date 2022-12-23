import numpy as np
import cv2, time
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP

imgname = "egon"

start = time.perf_counter()

file = open("{ing}.jhp".format(ing=imgname), "rb")

private_key = RSA.import_key(open("private.pem").read())

enc_session_key, nonce, tag, ciphertext = \
   [ file.read(x) for x in (private_key.size_in_bytes(), 16, 16, -1) ]

file.close()

cipher_rsa = PKCS1_OAEP.new(private_key)
session_key = cipher_rsa.decrypt(enc_session_key)
cipher_aes = AES.new(session_key, AES.MODE_EAX, nonce)
data = cipher_aes.decrypt_and_verify(ciphertext, tag)
content = data.decode("utf-8")

stmp = content[:12]
size = [int(stmp[:4]), int(stmp[4:8]), int(stmp[8:12])]
img = np.zeros((size[0], size[1], size[2]), dtype=np.uint8)

sp = 12
for i in range(size[0]):
    for j in range(size[1]):
        img[i][j] = [int(content[sp:sp+3]), int(content[sp+3:sp+6]), int(content[sp+6:sp+9])]
        sp += 9

cv2.imshow("Press c to terminate", img)

end = time.perf_counter()
print("Time taken: {t}".format(t=end-start))

cv2.waitKey(0)
cv2.destroyAllWindows()