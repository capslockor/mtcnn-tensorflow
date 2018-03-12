from Core.Mtcnn import MtcnnDetector


pwf = './Weight/pnet.npy'
rwf = './Weight/rnet.npy'
owf = './Weight/onet.npy'

detector = MtcnnDetector(pwf, rwf, owf)

im = './1.jpg'

detector.detectImage(im, vis=True)

