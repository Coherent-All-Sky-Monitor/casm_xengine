import os
import time
import re
import subprocess
import statistics
from datetime import datetime

# Configuration
NPACKETS_PER_BLOCK = 2048
NANTS = 256
NBEAMS = 256
NCHAN_PER_PACKET = 512
SAMPLING_TIME_US = 12  # microseconds per sample

# Expected data production time per block (in seconds)
EXPECTED_BLOCK_TIME = (NPACKETS_PER_BLOCK * SAMPLING_TIME_US) / 1_000_000

print("=== CASM Beamformer Benchmark ===")
print(f"Configuration: {NANTS} ants, {NBEAMS} beams")
print(f"Expected block time: {EXPECTED_BLOCK_TIME:.6f} seconds")
print()

# Clean up existing databases
os.system("dada_db -k daaa -d")
os.system("dada_db -k dddd -d")

# Create databases (exact working commands)
print("Creating DADA databases...")
os.system("dada_db -k daaa -b 1073741824 -n 4")
os.system("dada_db -k dddd -b 16777216 -n 4")

# Start fake_writer (exact working command)
print("Starting fake_writer...")
os.system("./fake_writer &")
time.sleep(2)

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

# Run beamformer (exact working command)
cmd = "./casm_bfCorr -b -i daaa -o dddd -f empty.flags -a dummy.calib -p powers.out"
process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Parse timing output
line_count = 0
start_time = time.time()
timeout_seconds = 60  # 1 minute timeout

for line in process.stderr:
    # Check for timeout
    if time.time() - start_time > timeout_seconds:
        print(f"\n⚠️  Timeout reached ({timeout_seconds}s). Stopping benchmark.")
        process.terminate()
        break
    if "spent time" in line:
        line_count += 1
        
        # Parse timing values
        match = re.search(r'spent time ([\d.e+-]+) ([\d.e+-]+) ([\d.e+-]+) ([\d.e+-]+) s', line)
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
                print(f"Processed {line_count} blocks... (RT ratio: {real_time_ratio:.3f})")
        
        print(line.strip())

# Wait for process to complete
process.wait()
end_time = time.time()

# Calculate summary statistics
if total_times:
    print("\n" + "=" * 50)
    print("=== PERFORMANCE SUMMARY ===")
    print(f"Total blocks processed: {len(total_times)}")
    print(f"Total runtime: {end_time - start_time:.2f} seconds")
    print(f"Average blocks per second: {len(total_times) / (end_time - start_time):.2f}")
    
    # Timing statistics
    print("\n--- TIMING BREAKDOWN ---")
    print(f"Copy time:     {statistics.mean(copy_times):.6f} ± {statistics.stdev(copy_times):.6f} s")
    print(f"Prep time:     {statistics.mean(prep_times):.6f} ± {statistics.stdev(prep_times):.6f} s")
    print(f"CUBLAS time:   {statistics.mean(cublas_times):.6f} ± {statistics.stdev(cublas_times):.6f} s")
    print(f"Output time:   {statistics.mean(output_times):.6f} ± {statistics.stdev(output_times):.6f} s")
    print(f"Total time:    {statistics.mean(total_times):.6f} ± {statistics.stdev(total_times):.6f} s")
    
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
        print(f"✅ REAL-TIME ACHIEVED (margin: {(avg_rt_ratio-1)*100:.1f}%)")
    else:
        print(f"❌ NOT REAL-TIME (deficit: {(1-avg_rt_ratio)*100:.1f}%)")
    
    # Throughput metrics
    print("\n--- THROUGHPUT METRICS ---")
    packets_per_second = (NPACKETS_PER_BLOCK * len(total_times)) / (end_time - start_time)
    print(f"Packets processed per second: {packets_per_second:.0f}")
    print(f"Beams per second: {NBEAMS * len(total_times) / (end_time - start_time):.0f}")
    print(f"Antennas × Beams per second: {NANTS * NBEAMS * len(total_times) / (end_time - start_time):.0f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmark_results_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("CASM Beamformer Benchmark Results\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Configuration: {NANTS} ants, {NBEAMS} beams\n")
        f.write(f"Total blocks: {len(total_times)}\n")
        f.write(f"Total runtime: {end_time - start_time:.2f} seconds\n")
        f.write(f"Average real-time ratio: {avg_rt_ratio:.3f}\n")
        f.write(f"Average total time per block: {avg_total:.6f} seconds\n")
        f.write(f"Copy time: {statistics.mean(copy_times):.6f} s ({statistics.mean(copy_times)/avg_total*100:.1f}%)\n")
        f.write(f"Prep time: {statistics.mean(prep_times):.6f} s ({statistics.mean(prep_times)/avg_total*100:.1f}%)\n")
        f.write(f"CUBLAS time: {statistics.mean(cublas_times):.6f} s ({statistics.mean(cublas_times)/avg_total*100:.1f}%)\n")
        f.write(f"Output time: {statistics.mean(output_times):.6f} s ({statistics.mean(output_times)/avg_total*100:.1f}%)\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    
else:
    print("No timing data collected!")

# Clean up
os.system("pkill -f fake_writer")

print("\nBenchmark completed.")
