Set objFSO = CreateObject("Scripting.FileSystemObject")
Set objShell = CreateObject("WScript.Shell")
strFolder = objFSO.GetParentFolderName(WScript.ScriptFullName)
objShell.Run """" & strFolder & "\launch.bat""", 0, False
