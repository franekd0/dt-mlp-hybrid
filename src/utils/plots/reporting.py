def print_results(time, accuracy):
    print(f"-> Accuracy: {accuracy:.4f}\n-> Time: {time:.4f}s")


def print_table(labels, accuracies, times):
    print("\n" + "=" * 110)
    print(f"{'Model':<70} | {'Accuracy (mean ± std)':<22} | {'Time [s]':<10}")
    print("-" * 110)
    for i in range(len(labels)):
        print(f"{labels[i]:<70} | {accuracies[i]:<22} | {times[i]:<10}")
    print("=" * 110)