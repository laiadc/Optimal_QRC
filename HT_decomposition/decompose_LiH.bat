@echo off

FOR /F "delims=," %%x in (angles_LiH_new_params.csv) DO (
	cd C:\cabal\bin
	echo %%x>> %1
	gridsynth %%x>> %1
)
PAUSE