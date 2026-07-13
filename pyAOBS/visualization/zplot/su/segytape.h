typedef struct {
long    job_no;    /* profileNo x1000 + stationNo */
long    line_no;   /* profile No */
char	dum1[8];
short	samp_interval;
short	dum3;
short	num_samples_pertrace;
short   dum4;
short   dat_format_code;
char	dum2[374];
}reelhd;
