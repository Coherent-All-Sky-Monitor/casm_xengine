import os
import time
import re
import subprocess
import statistics
import signal
import psutil
from datetime import datetime

junkdb = False
dir = '/home/user/software/casm_xengine/src'
in_key = 'daaa'
out_key = 'dddd'
in_block_size = 1073741824
out_block_size = 16777216

# Constants for calculations
NPACKETS_PER_BLOCK = 2048
NANTS = 256
NBEAMS = 1024
NCHAN_PER_PACKET = 512
SAMPLING_TIME_US = 12  # microseconds per sample

# Expected data production time per block (in seconds)
EXPECTED_BLOCK_TIME = (NPACKETS_PER_BLOCK * SAMPLING_TIME_US) / 1_000_000

print("=== CASM Beamformer Benchmark ===")
print("Configuration:")
print(f"  - Antennas: {NANTS}")
print(f"  - Beams: {NBEAMS}")
print(f"  - Packets per block: {NPACKETS_PER_BLOCK}")
print(f"  - Sampling time: {SAMPLING_TIME_US} μs")
print(f"  - Expected block time: {EXPECTED_BLOCK_TIME:.6f} seconds")
print("  - Target real-time ratio: 1.0")
print()

# Clean up existing databases
try:
    os.system(f"dada_db -k {in_key} -d")
    os.system(f"dada_db -k {out_key} -d")
except Exception:
    pass

# Create new databases
print("Creating DADA databases...")
os.system(f"dada_db -k {in_key} -b {in_block_size} -n 4")
os.system(f"dada_db -k {out_key} -b {out_block_size} -n 4")
time.sleep(1)

# Start data generation
print("Starting data generation...")
if junkdb:
    os.system(f"dada_junkdb -k {in_key} -t 3600 header.txt")
else:
    # Start fake_writer in background and capture its PID
    fake_writer_cmd = f"{dir}/fake_writer"
    fake_writer_process = subprocess.Popen(fake_writer_cmd, shell=True)
    print(f"Started fake_writer with PID: {fake_writer_process.pid}")
    
    # Check if fake_writer is running
    time.sleep(2)
    try:
        # Check if process is still running
        if fake_writer_process.poll() is None:
            print("✅ fake_writer is running")
        else:
            print("❌ fake_writer has stopped")
            print(f"Return code: {fake_writer_process.returncode}")
    except Exception as e:
        print(f"⚠️  Error checking fake_writer status: {e}")

time.sleep(2)  # Give more time for processes to start

# Check DADA database status
print("\nChecking DADA database status...")
os.system(f"dada_db -k {in_key} -s")
os.system(f"dada_db -k {out_key} -s")
print()

# Start beamformer and capture output
print("Starting beamformer...")
print("=" * 50)

# Lists to store timing data
copy_times = []
prep_times = []
cublas_times = []
output_times = []
total_times = []
real_time_ratios = []

# Run beamformer and capture output
cmd = (f"{dir}/casm_bfCorr -b -i {in_key} -o {out_key} "
       f"-f {dir}/empty.flags -a {dir}/dummy.calib -p {dir}/powers.out")
process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True)

# Parse timing output with timeout
line_count = 0
start_time = time.time()
timeout_seconds = 60  # 1 minute timeout
last_output_time = time.time()

print("Monitoring beamformer output...")
print("(Press Ctrl+C to stop early)")

try:
    for line in process.stderr:
        current_time = time.time()
        last_output_time = current_time
        
        # Check for timeout
        if current_time - start_time > timeout_seconds:
            print(f"\n⚠️  Timeout reached ({timeout_seconds}s). Stopping benchmark.")
            break
            
        if "spent time" in line:
            line_count += 1
            
            # Parse timing values
            match = re.search(r'spent time ([\d.e+-]+) ([\d.e+-]+) '
                             r'([\d.e+-]+) ([\d.e+-]+) s', line)
            if match:
                copy_time = float(match.group(1))
                prep_time = float(match.group(2))
                cublas_time = float(match.group(3))
                output_time = float(match.group(4))
                
                total_time = copy_time + prep_time + cublas_time + output_time
                real_time_ratio = EXPECTED_BLOCK_TIME / total_time
                
                # Store values
                copy_times.append(copy_time)
                prep_times.append(prep_time)
                cublas_times.append(cublas_time)
                output_times.append(output_time)
                total_times.append(total_time)
                real_time_ratios.append(real_time_ratio)
                
                # Print progress every 10 blocks
                if line_count % 10 == 0:
                    print(f"Processed {line_count} blocks... "
                          f"(RT ratio: {real_time_ratio:.3f})")
            
            print(line.strip())
        else:
            # Print other output for debugging
            if "DEBUG:" in line or "ERROR:" in line or "WARNING:" in line:
                print(f"[BEAMFORMER] {line.strip()}")
            elif line.strip():  # Print non-empty lines
                print(f"[BEAMFORMER] {line.strip()}")

except KeyboardInterrupt:
    print("\n⚠️  Benchmark interrupted by user.")
    print("Cleaning up processes...")

finally:
    # Clean up processes
    print("Cleaning up...")
    
    # Kill beamformer process
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    
    # Kill fake_writer if it's still running
    if not junkdb and 'fake_writer_process' in locals():
        try:
            fake_writer_process.terminate()
            fake_writer_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            fake_writer_process.kill()
    
    # Kill any remaining fake_writer processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'fake_writer' in str(proc.info['cmdline']):
                print(f"Killing fake_writer process {proc.info['pid']}")
                proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

end_time = time.time()

# Calculate summary statistics
if total_times:
    print("\n" + "=" * 50)
    print("=== PERFORMANCE SUMMARY ===")
    print(f"Total blocks processed: {len(total_times)}")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    print(f"Average blocks per second: "
          f"{len(total_times) / (end_time - start_time):.2f}")
    
    # Timing statistics
    print("\n--- TIMING BREAKDOWN ---")
    print(f"Copy time:     {statistics.mean(copy_times):.6f} ± "
          f"{statistics.stdev(copy_times):.6f} s")
    print(f"Prep time:     {statistics.mean(prep_times):.6f} ± "
          f"{statistics.stdev(prep_times):.6f} s")
    print(f"CUBLAS time:   {statistics.mean(cublas_times):.6f} ± "
          f"{statistics.stdev(cublas_times):.6f} s")
    print(f"Output time:   {statistics.mean(output_times):.6f} ± "
          f"{statistics.stdev(output_times):.6f} s")
    print(f"Total time:    {statistics.mean(total_times):.6f} ± "
          f"{statistics.stdev(total_times):.6f} s")
    
    # Percentage breakdown
    avg_total = statistics.mean(total_times)
    print("\n--- PERCENTAGE BREAKDOWN ---")
    print(f"Copy:     {statistics.mean(copy_times)/avg_total*100:.1f}%")
    print(f"Prep:     {statistics.mean(prep_times)/avg_total*100:.1f}%")
    print(f"CUBLAS:   {statistics.mean(cublas_times)/avg_total*100:.1f}%")
    print(f"Output:   {statistics.mean(output_times)/avg_total*100:.1f}%")
    
    # Real-time performance
    print("\n--- REAL-TIME PERFORMANCE ---")
    avg_rt_ratio = statistics.mean(real_time_ratios)
    min_rt_ratio = min(real_time_ratios)
    max_rt_ratio = max(real_time_ratios)
    print(f"Average real-time ratio: {avg_rt_ratio:.3f}")
    print(f"Min real-time ratio:     {min_rt_ratio:.3f}")
    print(f"Max real-time ratio:     {max_rt_ratio:.3f}")
    
    if avg_rt_ratio >= 1.0:
        print(f"✅ REAL-TIME ACHIEVED (margin: "
              f"{(avg_rt_ratio-1)*100:.1f}%)")
    else:
        print(f"❌ NOT REAL-TIME (deficit: "
              f"{(1-avg_rt_ratio)*100:.1f}%)")
    
    # Throughput metrics
    print("\n--- THROUGHPUT METRICS ---")
    packets_per_second = (NPACKETS_PER_BLOCK * len(total_times)) / (end_time - start_time)
    print(f"Packets processed per second: {packets_per_second:.0f}")
    print(f"Beams per second: "
          f"{NBEAMS * len(total_times) / (end_time - start_time):.0f}")
    print(f"Antennas × Beams per second: "
          f"{NANTS * NBEAMS * len(total_times) / (end_time - start_time):.0f}")
    
    # Memory and computational intensity
    print("\n--- COMPUTATIONAL INTENSITY ---")
    total_ops = len(total_times) * NANTS * NBEAMS * NCHAN_PER_PACKET * 8
    ops_per_second = total_ops / (end_time - start_time)
    print(f"Estimated operations per second: {ops_per_second:.2e}")
    
    # Save detailed results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmark_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("CASM Beamformer Benchmark Results\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Configuration: {NANTS} ants, {NBEAMS} beams, "
                f"{NPACKETS_PER_BLOCK} packets/block\n")
        f.write(f"Total blocks: {len(total_times)}\n")
        f.write(f"Total runtime: {end_time - start_time:.2f} seconds\n")
        f.write(f"Average real-time ratio: {avg_rt_ratio:.3f}\n")
        f.write(f"Average total time per block: {avg_total:.6f} seconds\n")
        f.write(f"Copy time: {statistics.mean(copy_times):.6f} s "
                f"({statistics.mean(copy_times)/avg_total*100:.1f}%)\n")
        f.write(f"Prep time: {statistics.mean(prep_times):.6f} s "
                f"({statistics.mean(prep_times)/avg_total*100:.1f}%)\n")
        f.write(f"CUBLAS time: {statistics.mean(cublas_times):.6f} s "
                f"({statistics.mean(cublas_times)/avg_total*100:.1f}%)\n")
        f.write(f"Output time: {statistics.mean(output_times):.6f} s "
                f"({statistics.mean(output_times)/avg_total*100:.1f}%)\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    
else:
    print("No timing data collected!")
    print("This might indicate:")
    print("1. The beamformer didn't start properly")
    print("2. There's a data format issue")
    print("3. The fake_writer isn't producing data")
    print("4. There's a synchronization issue")

print("\nBenchmark completed.")
