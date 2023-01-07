drop database if exists crop_recommendation;
create database if not exists crop_recommendation;
use crop_recommendation;


create table predicted_output(
N int not null,
P int not null,
K int not null,
temperature float not null,
humidity float not null,
ph float not null,
rainfall float not null,
prediction varchar(50) DEFAULT null);

select * from predicted_output;

select 
floor(avg(N)) avg_N,
floor(avg(P)) avg_P, 
floor(avg(K)) avg_K, 
prediction
from predicted_output
group by prediction
order by 1, 2, 3;
