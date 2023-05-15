# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import lander

var_glob = lander.GlobalVar()

state = [1000, 100, 0]
new_state = lander.lander_model(state, 100, var_glob)
print("state = "+str(new_state))
