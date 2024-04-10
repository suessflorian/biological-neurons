def printf(string):
    # function for printing stuff that gets removed from the output each iteration
    import sys
    from IPython.display import clear_output
    clear_output(wait = True)
    print(string, end = '')
    sys.stdout.flush()