
# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#from Generar_Resumen_api import *


from Generar_Resumen_api import main

def summarize(file_path, percentile):
    return main(file_path, percentile)



if __name__ == "__main__":
    main()
