from loader_test import run_loader_test
from chunk_test import run_chunk_size_test, run_overlap_test
from retrieval_test import run_topk_test
from generation_test import run_temperature_test
from metrics import print_title


def main():

    while True:

        print_title("RAG Evaluation")

        print("1 - Dokumentloader testen")
        print("2 - Chunk Size testen")
        print("3 - Overlap testen")
        print("4 - Top-k testen")
        print("5 - Temperature testen")
        print("0 - Beenden")

        choice = input("\nAuswahl: ")

        if choice == "1":
            run_loader_test()

        elif choice == "2":
            run_chunk_size_test()

        elif choice == "3":
            run_overlap_test()

        elif choice == "4":
            run_topk_test()

        elif choice == "5":
            run_temperature_test()

        elif choice == "0":
            break

        else:
            print("Ungültige Eingabe.")


if __name__ == "__main__":
    main()