# 2023_Postal_codes_selector
The CodesSelector tool was developed for the purpose of finding the user chosen number of postal codes spread over the territory of a country in a regular way. The initial application of this tool was webscaping but it can serve as base for different location analysies based on postal codes or geocoordinates (latitude and longitude). The tool is using k-means algorithm to group the postal codes according to their location and then to select the most central ones for each group. The particular model can be saved but if there are a little bit different codes or geocoordinates needed at every run this can be achieved by just running the script without saving the model.
