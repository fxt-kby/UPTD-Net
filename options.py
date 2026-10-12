import argparse

parser = argparse.ArgumentParser(description="UPTD-Net")

parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--datasets', type=str, default='input/illu3/s0', help='S0 intensity images')
parser.add_argument('--datasets2', type=str, default='input/illu3/dop', help='degree-of-polarization maps')
parser.add_argument('--datasets3', type=str, default='input/illu3/s_perp', help='perpendicular polarization component')
parser.add_argument('--clip', type=bool, default=True)
parser.add_argument('--num_iter', type=int, default=301)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--learning_rate2', type=float, default=0.0001)

parser.add_argument('--output_folder', type=str, default='output')
parser.add_argument('--down_size', type=int, default=450, help='side length of the square input')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--window', type=int, default=5, help='context window size')
parser.add_argument('--learning_rate3', type=float, default=1e-5)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--beta', type=float, default=20)
parser.add_argument('--gamma', type=float, default=10)
parser.add_argument('--delta', type=float, default=5)
opt = parser.parse_args()
