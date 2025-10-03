#!/usr/bin/env python3
"""
PID Tuning Script for Self-Driving Car
This script uses the PID debug simulator to test and optimize PID parameters.
"""

import argparse
from beamng_sim.utils.pid_debug import run_pid_test, run_batch_pid_tests
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='PID Tuning Tool for Self-Driving Car')
    parser.add_argument('--batch', action='store_true', help='Run batch tests of multiple PID configurations')
    parser.add_argument('--kp', type=float, default=0.15, help='Proportional gain')
    parser.add_argument('--ki', type=float, default=0.005, help='Integral gain')
    parser.add_argument('--kd', type=float, default=0.1, help='Derivative gain')
    parser.add_argument('--noise', type=float, default=0.1, help='Simulation noise level (0.0-1.0)')
    parser.add_argument('--curves', type=float, default=0.3, help='Curve difficulty (0.0-1.0)')
    parser.add_argument('--speed', type=float, default=40.0, help='Vehicle speed in km/h')
    parser.add_argument('--steps', type=int, default=500, help='Simulation steps')
    args = parser.parse_args()
    
    if args.batch:
        print("Running batch PID tests to compare multiple configurations...")
        best_config = run_batch_pid_tests()
        print("\nRecommended PID values for your self-driving car:")
        print(f"Kp = {best_config['Kp']}")
        print(f"Ki = {best_config['Ki']}")
        print(f"Kd = {best_config['Kd']}")
    else:
        print(f"Testing PID with Kp={args.kp}, Ki={args.ki}, Kd={args.kd}")
        sim_params = {
            'sim_length': args.steps,
            'noise_level': args.noise,
            'curve_difficulty': args.curves,
            'speed_kph': args.speed
        }
        metrics = run_pid_test(args.kp, args.ki, args.kd, **sim_params)
    
    print("\n=== WHAT TO LOOK FOR IN THE PLOTS ===")
    print("1. DEVIATION PLOT (top):")
    print("   - Should oscillate minimally around zero")
    print("   - Should quickly return to zero after curves")
    print("   - Smaller amplitude oscillations = better")
    
    print("\n2. STEERING PLOT (middle):")
    print("   - Should be smooth, not jagged")
    print("   - Should not oscillate rapidly back and forth")
    print("   - Should show appropriate response to curves")
    
    print("\n3. POSITION PLOT (bottom):")
    print("   - Blue line (vehicle) should closely follow green line (lane)")
    print("   - Vehicle line should be smooth, not zigzagging")
    
    print("\n=== TUNING GUIDE ===")
    print("ISSUE: Car zigzags too much (oscillation)")
    print("FIX:   - Decrease Kp (reduces overreaction)")
    print("       - Increase Kd (adds more damping)")
    print("       - Lower Ki further (reduces oscillation buildup)")
    
    print("\nISSUE: Car responds too slowly")
    print("FIX:   - Increase Kp carefully")
    print("       - Decrease Kd slightly")
    
    print("\nISSUE: Car doesn't center in lane")
    print("FIX:   - Increase Ki slightly (improves steady-state)")
    
    print("\n=== RECOMMENDED STARTING VALUES ===")
    print("- For stable driving:      Kp=0.15, Ki=0.005, Kd=0.1")
    print("- For smoother driving:    Kp=0.1, Ki=0.001, Kd=0.15")
    print("- For zigzag elimination:  Kp=0.08, Ki=0.001, Kd=0.2")
    
    print("\n=== SYSTEMATIC TUNING PROCESS ===")
    print("1. Start with Kp only (Ki=0, Kd=0)")
    print("2. Increase Kp until you see slight oscillation, then reduce by 20%")
    print("3. Add Kd to eliminate oscillations")
    print("4. Add small Ki (0.001-0.005) to improve centering")

if __name__ == "__main__":
    main()