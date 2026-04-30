# Experiment 6: Qualitative Analysis of Model Behaviour

This qualitative analysis uses existing experiment outputs only. It compares baseline and best prompt outputs to show how unnecessary edits appear in practice.

- Baseline prompt: `baseline`
- Best prompt: `role`

## Example 1: Qualitative difference

**Original:**
One day I visit my friend Jimmy in New York city , he was a young man who was specialist on trains and tourism , he talked about how the citizens and the commuters move from one place to another , he told me that the grand central terminal was the largest terminal of the city , he show me were the landmarks of the big apple were so sightseers could go , he show me the city and we went to different parts first he took me to Columbus circle in the south west corner of central park were there were the most expensive apartments then we went to the lake were the jogging tracks that circle the lake area popular with early morning visitors , then we went to the museum of natural history that was located near the metropolitan museum of art then I got focal on the subway trains so we went to Grand central terminal when we arrive I was amazed to see many people going to work so he told me that it was conveniently for people to use the train because is very fastest and for the government it was a grat economic business then he told me that one of the characteristics of

**Reference:**
One day I visited my friend Jimmy in New York city . He was a young man who was an expert on trains and tourism . He talked about how citizens and commuters move from one place to another . He told me that Grand Central Station was the largest terminus in the city . He showed me where the landmarks of the Big Apple were so sightseers could go there . He showed me the city and we went to different parts . First he took me to Columbus Circle in the south west corner of Central Park where there were the most expensive apartments . Then we went to the lake where the jogging tracks that circle the lake are popular with early - morning visitors . Then we went to the Museum of Natural History that was located near the Metropolitan Museum of Art . Then I got focal on the subway trains , so we went to Grand Central Station . When we arrived , I was amazed to see a lot of people going to work , so he told me that it was convenient for people to use the train because it is very fast and for the government it was a great economic business . Then he told me that one of the characteristics of

**Baseline output:**
One day, I visited my friend Jimmy in New York City. He was a young man who was a specialist in trains and tourism. He talked about how the citizens and commuters move from one place to another. He told me that the Grand Central Terminal was the largest terminal in the city. He showed me where the landmarks of the Big Apple were so sightseers could go. He showed me the city, and we went to different parts. First, he took me to Columbus Circle in the southwest corner of Central Park, where there were the most expensive apartments. Then we went to the lake, where the jogging tracks that circle the lake

**Best prompt output:**
One day I visited my friend Jimmy in New York City, he was a young man who was a specialist on trains and tourism, he talked about how the citizens and the commuters move from one place to another, he told me that the Grand Central Terminal was the largest terminal of the city, he showed me where the landmarks of the Big Apple were so sightseers could go, he showed me the city and we went to different parts first he took me to Columbus Circle in the southwest corner of Central Park where there were the most expensive apartments then we went to the lake where the jogging tracks that circle the lake are popular with early morning visitors

**Explanation:**
The baseline baseline edits the sentence more aggressively, and the baseline baseline deviates from the original wording more than the best prompt, and the baseline baseline and best prompt outputs differ in structure or wording. This suggests that the baseline introduces unnecessary edits while the best prompt preserves the sentence more closely.

**Metadata:**
- baseline_edit_distance: 133
- best_prompt_edit_distance: 113
- baseline_oci: 0.016784
- best_prompt_oci: 0.010496

## Example 2: Qualitative difference

**Original:**
Personal experiences

**Reference:**
Personal experiences

**Baseline output:**
I'm sorry, but I need the full sentence to provide a correction. Could you please provide the complete sentence?

**Best prompt output:**
Personal experience

**Explanation:**
The baseline baseline edits the sentence more aggressively, and the baseline baseline deviates from the original wording more than the best prompt, and the baseline baseline and best prompt outputs differ in structure or wording. This suggests that the baseline introduces unnecessary edits while the best prompt preserves the sentence more closely.

**Metadata:**
- baseline_edit_distance: 19
- best_prompt_edit_distance: 1
- baseline_oci: 0.016784
- best_prompt_oci: 0.010496

## Example 3: Qualitative difference

**Original:**
After walking for a lot of days , the ' survivors ' were worried , and thinking of how much they would still need to walk to find a city , also they were thinking that maybe they were going forward one day and backwards the next day .

**Reference:**
After walking for many days , the ' survivors ' were worried , and thinking about how much they would still need to walk to find a city . Also they were thinking that maybe they were going forward one day and backwards the next day .

**Baseline output:**
After walking for many days, the 'survivors' were worried and thinking about how much farther they would need to walk to find a city. They were also concerned that they might be going forward one day and backward the next.

**Best prompt output:**
After walking for a lot of days, the 'survivors' were worried and thinking of how much they would still need to walk to find a city; also, they were thinking that maybe they were going forward one day and backward the next day.

**Explanation:**
The baseline baseline edits the sentence more aggressively, and the baseline baseline deviates from the original wording more than the best prompt, and the baseline baseline and best prompt outputs differ in structure or wording. This suggests that the baseline introduces unnecessary edits while the best prompt preserves the sentence more closely.

**Metadata:**
- baseline_edit_distance: 25
- best_prompt_edit_distance: 12
- baseline_oci: 0.016784
- best_prompt_oci: 0.010496

## Example 4: Qualitative difference

**Original:**
Ritesh

**Reference:**
Ritesh

**Baseline output:**
I'm sorry, but it seems like the sentence is incomplete. Could you please provide the full sentence for correction?

**Best prompt output:**
Ritesh is going to the market yesterday.

**Explanation:**
The baseline baseline edits the sentence more aggressively, and the baseline baseline deviates from the original wording more than the best prompt, and the baseline baseline and best prompt outputs differ in structure or wording. This suggests that the baseline introduces unnecessary edits while the best prompt preserves the sentence more closely.

**Metadata:**
- baseline_edit_distance: 19
- best_prompt_edit_distance: 6
- baseline_oci: 0.016784
- best_prompt_oci: 0.010496

## Example 5: Qualitative difference

**Original:**
Public transport is not convenient for most of the people because every person wants to reach on time at his place but using public transport it is very difficult to reach on time .

**Reference:**
Public transport is not convenient for most the people because every person wants to arrive at his destination on time , but when using public transport it is very difficult to arrive on time .

**Baseline output:**
Public transport is not convenient for most people because everyone wants to reach their destination on time, but using public transport makes it very difficult to do so.

**Best prompt output:**
Public transport is not convenient for most of the people because every person wants to reach on time at his place, but using public transport it is very difficult to reach on time.

**Explanation:**
The baseline baseline edits the sentence more aggressively, and the baseline baseline deviates from the original wording more than the best prompt, and the baseline baseline and best prompt outputs differ in structure or wording. This suggests that the baseline introduces unnecessary edits while the best prompt preserves the sentence more closely.

**Metadata:**
- baseline_edit_distance: 15
- best_prompt_edit_distance: 3
- baseline_oci: 0.016784
- best_prompt_oci: 0.010496

## Example 6: Qualitative difference

**Original:**
I always stay home and take care of them , I invent games for them to have fun and they also can learn with them , sometimes I take them to the sports club I work on , I 'm a swimming teacher and on my freetimes on the afternoon I 'm with them in the pool having fun , playing and teaching them to swim .

**Reference:**
I always stay home and take care of them . I invent games for them to have fun and they also can learn with them . Sometimes I take them to the sports club I work at . I 'm a swimming teacher and in my free time in the afternoon , I 'm with them in the pool having fun , playing and teaching them to swim .

**Baseline output:**
I always stay home and take care of them. I invent games for them to have fun and learn from. Sometimes, I take them to the sports club where I work. I'm a swimming teacher, and in my free time in the afternoon, I'm with them in the pool, having fun, playing, and teaching them to swim.

**Best prompt output:**
I always stay home and take care of them, I invent games for them to have fun and they also can learn with them, sometimes I take them to the sports club I work at, I'm a swimming teacher and in my free time in the afternoon I'm with them in the pool having fun, playing and teaching them to swim.

**Explanation:**
The baseline baseline edits the sentence more aggressively, and the baseline baseline deviates from the original wording more than the best prompt, and the baseline baseline and best prompt outputs differ in structure or wording. This suggests that the baseline introduces unnecessary edits while the best prompt preserves the sentence more closely.

**Metadata:**
- baseline_edit_distance: 29
- best_prompt_edit_distance: 18
- baseline_oci: 0.016784
- best_prompt_oci: 0.010496

## Example 7: Qualitative difference

**Original:**
Lastly , workshops about conservation of environment to increase environmental awareness among people to encourage them to take care of their city , and how they can recycle some rubbish to beneficial objects to exploit it at home .

**Reference:**
Lastly , workshops about conservation of the environment to increase environmental awareness among people , to encourage them to take care of their city , and teach them how they can recycle some rubbish to make useful objects to use at home .

**Baseline output:**
Lastly, workshops about environmental conservation aim to increase awareness among people, encouraging them to take care of their city and teaching them how to recycle some rubbish into beneficial objects for use at home.

**Best prompt output:**
Lastly, workshops about conservation of the environment to increase environmental awareness among people to encourage them to take care of their city, and how they can recycle some rubbish into beneficial objects to exploit at home.

**Explanation:**
The baseline baseline edits the sentence more aggressively, and the baseline baseline deviates from the original wording more than the best prompt, and the baseline baseline and best prompt outputs differ in structure or wording. This suggests that the baseline introduces unnecessary edits while the best prompt preserves the sentence more closely.

**Metadata:**
- baseline_edit_distance: 20
- best_prompt_edit_distance: 9
- baseline_oci: 0.016784
- best_prompt_oci: 0.010496

## Example 8: Qualitative difference

**Original:**
Angel

**Reference:**
Angel

**Baseline output:**
I'm sorry, but it seems like the sentence you provided is incomplete. Could you please provide the full sentence for correction?

**Best prompt output:**
Angel is a good friend who always helps me with my homework.

**Explanation:**
The baseline baseline edits the sentence more aggressively, and the baseline baseline deviates from the original wording more than the best prompt, and the baseline baseline and best prompt outputs differ in structure or wording. This suggests that the baseline introduces unnecessary edits while the best prompt preserves the sentence more closely.

**Metadata:**
- baseline_edit_distance: 21
- best_prompt_edit_distance: 11
- baseline_oci: 0.016784
- best_prompt_oci: 0.010496
