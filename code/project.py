import argparse
from interface import app

if __name__ == '__main__':
    app.run_server(debug=True)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--year", type=int, default=2019)
    # parser.add_argument("--analyze", default="False")
    # args = parser.parse_args()

    # if args.analyze:
    #     print('\nGenerating Graph \n')
    # else:
    #     raise ValueError("You must enter a parameter to analyze, --analyze Author/Rank/Position/Area")

