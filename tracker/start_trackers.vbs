' start_trackers.vbs - the tracker, hidden: http://127.0.0.1:8765
' tracker_server.py keeps itself and the three orchestrators running (orchestrators.py), so it is all that needs starting.
' Run at login from the Startup folder; running it again is harmless (a second copy exits at once).
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "C:\Users\propr\Desktop\whiteboard\note_comprehension\tracker"
sh.Run "cmd /c python -u tracker_server.py >> tracker_server.log 2>&1", 0, False
