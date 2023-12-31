"""
This is the main module responsible for solving the tasks.
To solve each task just run `python main.py`.
"""

from A import A1
from A import A2
from B import B1
from B import B2


if __name__ == "__main__":
    """
    Main function to run the whole project
    It will run the process from A1 to A2 to B1 to B2
    if you want to change the order or run some of the file
    please just omment the file you do not want, for example:
    #A1.demonstration() will stop the running of A1
    """

    print("Running A1:")
    A1.demonstration()
    print("\n")
    print("Running A2:")
    A2.demonstration()
    print("\n")
    print("Running B1:")
    B1.demonstration()
    print("\n")
    print("Running B2:")
    B2.demonstration()
