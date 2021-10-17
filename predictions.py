import shutil
import pandas as pd
import os

df = pd.read_csv('results/results.csv')

for i in df.index:
    source = df.loc[i, 'filename']
    print('source : ', source)
    repertoire_destination = 'predictions/' + str(df.loc[i, 'pred_labels'])
    if not os.path.exists(repertoire_destination): os.makedirs(repertoire_destination)

    try:
        shutil.copy2(source, repertoire_destination)
        print("File copied successfully.")
    # Source = destination
    except shutil.SameFileError:
        print("Source = destination")
    # Problème de droits
    except PermissionError:
        print("Problème de droits")
    # For other errors
    except:
        print("Autre erreur")
