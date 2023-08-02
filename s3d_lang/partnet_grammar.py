import executor as ex

PROGS = {}
COUNT = [0]

# Treat shapes in file as a 'grammar' to sample from in ShapeCoder

def load_progs(data_path):
    cnt = 0
    with open(f'../s3d_lang/prog_data/{data_path}.txt') as f:
        for line in f:
            prog = line.split(':')[1].strip()
            PROGS[cnt] = prog
            cnt += 1


def sample_shape_program():
    cnt = COUNT[0]
    if cnt not in PROGS:
        assert False, ' sampled too many progs'

    prog_text = PROGS[cnt]
    COUNT[0] += 1

    P = ex.Program()
    P.run(prog_text)

    return P
