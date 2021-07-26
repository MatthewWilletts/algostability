import os


def get_doc(doc, baseline, augment, positive, feature_size, arch):
    if baseline:
        doc += 'Baseline'
    if feature_size > 0:
        doc += str(feature_size)
    doc += arch.lower()
    return doc


def check_rep(dset, baseline=False, augment=False, positive=False, feature_size=0, arch='ConvMLP'):
    doc = get_doc('representation', baseline, augment, positive, feature_size, arch)
    path = os.path.join('run', 'checkpoints', dset, doc)
    files = os.listdir(path)
    acc1 = []
    acc2 = []
    for i in range(20):
        if 'seed{}'.format(i) not in files:
            acc1.append(i)
    for f in files:
        if 'test_representations.p' not in os.listdir(os.path.join(path, f)):
            acc2.append(f)
    if len(acc1) > 0 or len(acc2) > 0:
        print('\n[REP] d: {}; b: {}; a: {}; p: {}; f: {}'.format(dset, baseline, augment, positive, feature_size))
        if len(acc1) > 0:
            print('Representations missing for seeds {}'.format(acc1))
        if len(acc2) > 0:
            print('Representations coorupted for {}'.format(acc2))

