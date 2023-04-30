#form Feature Extraction
#    word in_file 
#endform

#############
# Load file #
#############

#Read from file... 'in_file$'
Rename... sound
dur = Get total duration

#########
# Pitch #
#########

select Sound sound
To Pitch... 0 75 600

f0_mean = Get mean... 0 0 hertz
f0_pct5 = Get quantile... 0 0 0.05 hertz
f0_pct95 = Get quantile... 0 0 0.95 hertz
f0_range = f0_pct95 - f0_pct5

f0_mean_logHertz = Get mean... 0 0 logHertz
f0_pct5_logHertz = Get quantile... 0 0 0.05 logHertz
f0_pct95_logHertz = Get quantile... 0 0 0.95 logHertz
f0_range_logHertz = f0_pct95_logHertz - f0_pct5_logHertz

select Pitch sound
Remove

#############
# Intensity #
#############

select Sound sound
if dur > 6.4 / 100.0
    To Intensity... 100 0 no
    int_mean = Get mean... 0 0 energy
endif

#######
# NHR #
#######

select Sound sound
To Pitch... 0 75 600
To PointProcess
plus Sound sound
plus Pitch sound

voice_report$ = Voice report... 0 0 75.0 600.0 1.3 1.6 0.03 0.45
nhr = extractNumber(voice_report$, "Mean noise-to-harmonics ratio: ")
select Pitch sound
Remove
select Sound sound
To Pitch... 0 75 600

###########
# Voicing #
###########

vcd_frames = Count voiced frames
tot_frames = Get number of frames
vcd2tot_frames = vcd_frames / tot_frames

####################
# Jitter / Shimmer #
####################

if vcd_frames > 0 
	select Sound sound
	plus Pitch sound

	To PointProcess (cc)
    mean_period = 1 / f0_mean
	To TextGrid (vuv)... 0.02 mean_period

	select Sound sound
	plus TextGrid sound_sound
	Extract intervals... 1 no V
	Concatenate

	select Sound chain
    To Intensity... 100 0 no
    int_mean_vcd = Get mean... 0 0 energy
    select Sound chain


    select Sound chain
    To Pitch... 0 75 600
    To PointProcess
    plus Sound chain
    plus Pitch chain

    voice_report$ = Voice report... 0 0 75.0 600.0 1.3 1.6 0.03 0.45
    nhr_vcd = extractNumber(voice_report$, "Mean noise-to-harmonics ratio: ")
    select Pitch chain
    Remove


    select Sound chain
    dur_vcd = Get total duration
    if dur_vcd > (6.4 / 75)
        To Pitch... 0 75 600
        To PointProcess
        jitter = Get jitter (local)... 0 0 0.0001 0.02 1.3
        plus Sound chain
        shimmer = Get shimmer (local)... 0 0 0.0001 0.02 1.3 1.6
    endif

else
    dur_vcd = dur
    select PointProcess sound
    jitter = Get jitter (local)... 0 0 0.0001 0.02 1.3
    plus Sound sound
    shimmer = Get shimmer (local)... 0 0 0.0001 0.02 1.3 1.6
endif

##########
# Output #
##########

text$ = "duration,'dur'"
appendInfoLine: text$

text$ = "duration_vcd,'dur_vcd'"
appendInfoLine: text$

text$ = "pitch_mean,'f0_mean'"
appendInfoLine: text$

text$ = "pitch_5,'f0_pct5'"
appendInfoLine: text$

text$ = "pitch_95,'f0_pct95'"
appendInfoLine: text$

text$ = "pitch_range,'f0_range'"
appendInfoLine: text$

text$ = "pitch_mean_log,'f0_mean_logHertz'"
appendInfoLine: text$

text$ = "pitch_5_log,'f0_pct5_logHertz'"
appendInfoLine: text$

text$ = "pitch_95_log,'f0_pct95_logHertz'"
appendInfoLine: text$

text$ = "pitch_range_log,'f0_range_logHertz'"
appendInfoLine: text$

text$ = "intensity_mean,'int_mean'"
appendInfoLine: text$

text$ = "intensity_mean_vcd,'int_mean_vcd'"
appendInfoLine: text$

text$ = "jitter,'jitter'"
appendInfoLine: text$

text$ = "shimmer,'shimmer'"
appendInfoLine: text$

text$ = "nhr,'nhr'"
appendInfoLine: text$

text$ = "nhr_vcd,'nhr_vcd'"
appendInfoLine: text$