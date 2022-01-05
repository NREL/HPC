import datetime

from coad.COAD import COAD
from coad.ModelUtil import datetime_to_plex

coad = COAD('RTS-GMLC.xml')

#7/14/2024 
date_start = str(datetime_to_plex(datetime.datetime(2024, 7, 14, 0, 0)))
new_horizon = coad['Horizon']['Base'].copy("Interesting Week")
new_horizon["Step Count"] = "8"
new_horizon["Date From"] = date_start
new_horizon["Chrono Date From"] = date_start
new_horizon["Chrono Step Count"] = "8"
coad['Model']['DAY_AHEAD'].set_children(new_horizon)

coad.save("one_week_model.xml")
