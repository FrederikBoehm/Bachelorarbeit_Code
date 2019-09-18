mkdir -p "./data/Edgar/10-X_C";

# Load LM_10X_Summaries_2018.csv
curl "https://doc-0s-8s-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/sslu881qv00vjm962rjauqu2jhrcusr1/1568793600000/06691231387225241091/*/12YQ3bczd3-G94eSpqawbA1hwF0Jzs_jB?e=download" -o "./data/Edgar/LM_10X_Summaries_2018.csv";

# Load and unpack 10-X_C_2001-2005.zip, we delete the zip afterwards because we don't need it anymore
curl "https://doc-0c-8s-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/saeekfn7hi8adhub2jpfh6n7t7c9c1ss/1568793600000/06691231387225241091/*/11lR8DvQ7rxLTeyCr7-7JbPMRnpG3Bxme?e=download" -o "./data/Edgar/10-X_C_2001-2005.zip";
unzip "./data/Edgar/10-X_C_2001-2005.zip" -d "./data/Edgar/10-X_C";
rm "./data/Edgar/10-X_C_2001-2005.zip";

# Load and unpack 10-X_C_2006-2010.zip, we delete the zip afterwards because we don't need it anymore
curl "https://doc-0o-8s-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/0gmrnvhsl78jsbcn18e2t0qn5897nle9/1568793600000/06691231387225241091/*/124gT1LAPZGd4ngjGjVzQBPsBjn0fvFjy?e=download" -o "./data/Edgar/10-X_C_2006-2010.zip";
unzip "./data/Edgar/10-X_C_2006-2010.zip" -d "./data/Edgar/10-X_C";
rm "./data/Edgar/10-X_C_2006-2010.zip";

# Load and unpack 10-X_C_2011-2015.zip, we delete the zip afterwards because we don't need it anymore
curl "https://doc-0g-8s-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/7d4f4qpktnq4hp2mui05u2khs267jk70/1568793600000/06691231387225241091/*/11zcfTJHpNBicaBEmDDeYlrSUTPLfInuF?e=download" -o "./data/Edgar/10-X_C_2011-2015.zip";
unzip "./data/Edgar/10-X_C_2011-2015.zip" -d "./data/Edgar/10-X_C";
rm "./data/Edgar/10-X_C_2011-2015.zip";

# Load and unpack 10-X_C_2016-2018.zip, we delete the zip afterwards because we don't need it anymore
curl "https://doc-08-8s-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/r0btk3e8j7h1magb61368eqo1hrkde6i/1568793600000/06691231387225241091/*/11qyE9NKrVQR4sEY5t6ed3gKsCWo_QaRb?e=download" -o "./data/Edgar/10-X_C_2016-2018.zip";
unzip "./data/Edgar/10-X_C_2016-2018.zip" -d "./data/Edgar/10-X_C";
rm "./data/Edgar/10-X_C_2016-2018.zip";