# CtoF-ConUCB Codes for Reviewing

1. Run .sh file to get the results (remember to substitute the GPU indexes, plz see `scripts/task/*.sh`)


2. Algorithm name mapping:

   i. Env=Attr alg=LinUCB -> LinUCB 

   ii. Env=Attr alg=ArmCon -> ArmCon

   iii. Env=Attr alg=ConUCB -> ConUCB

   iv. Env=Kmeans alg=ConUCB -> CtoF-ConUCB

   v. Env=Kmeans alg=SKmeans -> CtoF-ConUCB-Clu

   vi. Env=Kmeans alg=Kmeans_user_ts_cluster -> CtoF-ConUCB-Clu+

3. Environment name mapping:

   i. data1_no_loc_tky_check_in (in codes) -> FourSquare

   ii. data2_delicious_KONECT (in codes) -> Delicious

   iii. data3_movielens_generator (in codes) -> MovieLens 25M

   iv. data4_lastfm (in codes) -> LastFM

   v. data5_bibsonomy (in codes) -> BibSonomy

   vi. data6_visualizeus (in codes) -> VisualizeUs

3. Dependences: environment.yml

4. And the use case study, cluster evaluation and semantic quality of generated key-terms are in `explore/` folder.
