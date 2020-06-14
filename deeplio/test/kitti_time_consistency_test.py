import argparse
import datetime as dt


def count_lines(path):
    with open(path, 'r') as f:
     length = len(f.readlines())
    return length


def run_test(args):
    try:
        ts_file = open(args['path'], 'r')
    except Exception as ex:
        print(ex)
        return
    print("Starting with {}".format(args['path']))

    n_lines = count_lines(args['path'])

    freq = float(args['freq'])
    thresh = float(args['thresh']) / 1000.
    period = 1. / freq

    last_time = None

    for i, line in enumerate(ts_file):
        time_curr = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')

        if last_time is None:
            last_time = time_curr
            continue

        t_diff = time_curr.timestamp() - last_time.timestamp()
        if t_diff < 0.:
            print("[{}] Negativ time diff: curr:{}, last:{}, diff:{:.4} [sec]".format(i, time_curr, last_time, 1000 * t_diff))

        if abs(t_diff) > (period + thresh):
            print("[{}] Exceeding time freq: curr:{}, last:{}, diff:{:.4} [ms]".format(i, time_curr, last_time, 1000 * abs(t_diff)))

        last_time = time_curr
        print("\r{:.3}%".format((i/n_lines) * 100.), end="",  flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to tiimestamps file", required=True)
    parser.add_argument("-f", "--freq", help="Expected sampling frequency in [hz]", required=True)
    parser.add_argument("-t", "--thresh", help="Allowed sampling time difference in [ms]", required=False, default=3)

    args = vars(parser.parse_args())
    run_test(args)