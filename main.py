import pandas as pd
import time
start_time = time.time()

#Enter your variables here
API = "lamivudine"
Solvent = "ethanol"
Target_GR = 0.01 #this is in um/s
Target_IT = 3600 #this is in s
Target_NR = 0.1 #this is in #/s


df = pd.read_excel(r'I:\cmac-lab-data\Crystalline\data\ThPi - Thomas Pickles\Experimental Master Sheet.xlsx', sheet_name= "Kinetics")
raw_df = df[(df["Solute"]==API) & (df["Solvent"]==Solvent)]

# Choose your algorithm here
from Algorithms import GA

print("--- %s seconds ---" % (time.time() - start_time))