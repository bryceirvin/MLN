import splitfolders
import sys

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("split_data.py <audio_dir> <save_dir> <train> <val> <test>")
        sys.exit()

    audio_dir = sys.argv[1]
    save_dir = sys.argv[2]
    train = float(sys.argv[3])
    val = sys.argv[4]
    test = sys.argv[5]

    splitfolders.ratio(audio_dir, output=save_dir, seed=69, ratio=(train, val, test))
    print("Split complete!")