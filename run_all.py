"""
Entry point: run both PINSTT case studies sequentially.

Usage:
    python run_all.py              # both
    python run_all.py --robot      # robot only
    python run_all.py --quadrotor  # quadrotor only
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot',     action='store_true')
    parser.add_argument('--quadrotor', action='store_true')
    args = parser.parse_args()

    run_robot     = args.robot or (not args.robot and not args.quadrotor)
    run_quadrotor = args.quadrotor or (not args.robot and not args.quadrotor)

    if run_robot:
        from experiments.robot_2d import run as run_robot_fn
        run_robot_fn()

    if run_quadrotor:
        from experiments.quadrotor_3d import run as run_quad_fn
        run_quad_fn()

    print("\nAll done. Check the plots/ folder for results.")


if __name__ == '__main__':
    main()
