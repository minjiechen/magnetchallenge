# MagNet Challenge 2023
## IEEE PELS-Google-Enphase-Princeton MagNet Challenge
<img src="img/mclogo.jpg" width="800">

## This site provides the latest information about the MagNet Challenge. 
## Please contact [pelsmagnet@gmail.com](mailto:pelsmagnet@gmail.com) for all purposes.
========================================================
## **MagNet Challenge 2023** [Final Evaluation Rules](finaltest/FinalEvaluationRules.pdf) Here:

On November 10th, 2023 - We have received 27 entries for the pre-test. If your team has submitted a pre-test report but was not labeled as [pretest] below, please let us know. Feel free to submit the results to conferences and journals, or seek IP protection. If you used MagNet data, please acknowledge MagNet by citing the papers listed at the end of this page.

On November 10th, 2023 – Data released for final evaluation:

1)	Download the new training data and testing data from the following link for 5 new materials similar or different from the previous 10 materials:
[MagNet Challenge Final Test Data](https://www.dropbox.com/sh/q5w2ddol8y6bk0k/AABXKxv_aiLj8yXspeusJq4na?dl=0)
2)	Train, tune, and refine your model or algorithm using the training data.
3)	Predict the core losses for all the data points contained in the testing data for the 5 materials. For each material, the prediction results should be formatted into a CSV file with a single column of core loss values. Please make sure the index of these values is consistent with the testing data, so that the evaluation can be conducted correctly.

On December 31st, 2023 – Final submission:
1)	Prediction results for the testing data are due as 5 separate CSV files for the 5 materials.
2)	For each material, package your best model as an executable MATLAB/Python function as P=function(B,T,f). This function should be able to directly read the original {B,T,f} CSV files and produce the predicted power P as a CSV file with a single column. 
3)	A 5-page IEEE TPEL format document due as a PDF file. Please briefly explain the key concepts.
4)	The authors listed on the 5-page report will be used as the final team member list.
5)	Report the total number of model parameters, as well as your model size as a table in the document. These numbers will be confirmed during the code review process.
6)	Full executable model due as a ZIP file for a potential code review with winning teams. These models should be fully executable on a regular personal computer without internet access after installing necessary packages. 
7)	Submit all the above required files to pelsmagnet@gmail.com.

January to March 2024 – Model Performance Evaluation, Code Review, Final Winner Selection:
1)  We will first evaluate the CSV core loss testing results for the 5 materials.
2)	10 to 15 teams with outstanding performance will be invited for a final code review with brief presentation.
3)	Evaluation criteria: high model accuracy; compact model size; good model readability.
4)	The final winners will be selected by the judging committee after jointly considering all the judging factors.
5)	All data, models, and results will be released to public, after the winners are selected.
6)	Our ultimate goal is to develop a "standard" datasheet model for each of the 15 materials.

========================================================
## **MagNet Challenge 2023** [Pretest Evaluation Rules](pretest/PreEvaluationRules.pdf) Here:

On November 10th, a preliminary test result is due to evaluate your already developed models for the 10 materials: 

- Step 1: Download the [MagNet Challenge Validation Data](https://www.dropbox.com/sh/4ppuzu7z4ky3m6l/AAApqXcxr_Fnr5x9f5qDr8j8a?dl=0) for the 10 existing materials each consisting of 5,000 randomly sampled data from the original database.

- Step 2: Use this database to evaluate your already-trained models.

- Step 3: Report your results following the provided [Template](pretest/PretestResultsPDF.pdf). Zip your Models and Results and send them to pelsmagnet@gmail.com.

We will use relative error to evaluate your models (the absolute error between the predicted and measured values).

$Percent\ Relative\  Error = \frac{\left |meas-pred \right |}{meas}\cdot100$ \%, where $meas$ is MagNet's Core Loss measurement and $pred$ is the model prediction.

The purpose of the preliminary test is to get you familiar with the final testing process. The preliminary test results have nothing to do with the final competition results. 

*** In the final test, we will provide a small or large dataset for training, and a small or large dataset for testing. The training and testing data for different materials may be offered in different ways to test the model's performance from different angles. ***

## MagNet Challenge Timeline

- 02-01-2023 MagNet Challenge Handbook Released [PDF](docs/handbook.pdf)
- 03-21-2023 Data Quality Report [PDF](docs/dataquality.pdf)
- 04-01-2023 Data for 10 Materials Available [Dropbox](https://www.dropbox.com/sh/yk3rsinvsj831a7/AAAC6vPwXSJgruxmq0EbNvrVa?dl=0)
- 05-15-2023 1-Page Letter of Intent Due with Signature [PDF](docs/registration.pdf) 
- 06-15-2023 2-Page Concept Proposal Due [PDF](docs/template.pdf) [DOC](docs/template.doc) [Latex](docs/ieeetran.zip)
- 07-01-2023 Notification of Acceptance (all 39 teams accepted)
- 08-01-2023 Expert Feedback on the Concept Proposal
- Teams develop a semi/fully-automated software pipeline to process data and generate models for 10 materials
- 11-10-2023 Preliminary Submission Due (postponed from 11-01-2023)
- Teams use the previously developed software pipeline to process new data and generate models for 3 new materials
- 12-31-2023 Final Submission Due (postponed from 12-24-2023)
- 03-01-2024 Winner Announcement and Presentation

## Evaluation Timeline

- 06-15-2023 Evaluate the concept proposals and ensure all teams understand the competition rules.
- 11-10-2023 Evaluate the 10 models the teams developed for the 10 materials and provide feedback for improvements.
- 12-31-2023 Evaluate the 3 new models the teams developed for the 3 new materials and announce the winners.

## Evaluation Criterias

The judging committee will evaluate the results of each team with the following criterias.
- Model accuracy (30%): core loss prediction accuracy evaluated by 95th percentile error (lower error better)
- Model size (30%): number of parameters the model needs to store for each material (smaller model better)
- Model explanability (20%): explanability of the model based on existing physical insights (more explainable better)
- Model novelty (10%): new concepts or insights presented by the model (newer insights better)
- Software quality (10%): quality of the software engineering (more concise better)

## MagNet Webinar Recordings

- 04-07-2023 MagNet Webinar Series #1 - Kickoff Meeting [Video](https://www.youtube.com/embed/vXiF10Ycqi4) [PDF](docs/webinar-1.pdf) 
- 05-12-2023 MagNet Webinar Series #2 - Equation-based Method [Video](https://www.youtube.com/watch?v=K1pZg0BAOss) [PDF](docs/webinar-2.pdf)
- 05-19-2023 MagNet Webinar Series #3 - Machine Learning Method [Video](https://www.youtube.com/watch?v=vEndPeBn6ng) [PDF](docs/webinar-3.pdf)
- 05-26-2023 MagNet Webinar Series #4 - Data Complexity and Quality [Video](https://www.youtube.com/watch?v=zU3B84H7aCU) [PDF](docs/webinar-4.pdf)

## MagNet Challenge Discussions

- MagNet GitHub Discussion Forum [Link](https://github.com/minjiechen/magnetchallenge/discussions)

## MagNet Baseline Tools and Tutorials

- MagNet: Equation-based Baseline Models - by Dr. Thomas Guillod (Dartmouth) [Link](https://github.com/otvam/magnet_webinar_eqn_models)
- MagNet: Machine Learning Tutorials - by Haoran Li (Princeton) [Link](https://github.com/minjiechen/magnetchallenge/tree/main/tutorials)
- MagNet: Data Processing Tools - by Dr. Diego Serrano (Princeton) [Link](https://github.com/minjiechen/magnetchallenge/tree/main/tools)

## MagNet Challenge Awards

- Model Performance Award, First Place        $10,000
- Model Performance Award, Second Place       $5,000
- Model Novelty Award, First Place            $10,000
- Model Novelty Award, Second Place           $5,000
- Outstanding Software Engineering Award      $5,000
- Honorable Mentions Award         multiple x $1,000

## Participating Teams (05-20-2023) 40 Teams from 18 Countries and Regions
## Denmark, USA, Brazil, China, India, Belgium, Spain, Singapore, Taiwan, Germany, Italy, South Korea, Austria, Nepal, Netherland, UK, Australia, Netherland
## All 39 Concept Papers have been Received !!!
## $30,000 budget from IEEE PELS confirmed!
## $10,000 gift from Google received!
## $10,000 gift from Enphase received!

- Aalborg University, Aalborg, Denmark 🇩🇰
- Arizona State University, Tempe AZ, USA 🇺🇸 - [pretest] 
- Cornell University Team 1, Ithaca, USA 🇺🇸
- Cornell University Team 2, Ithaca, USA 🇺🇸
- Federal University of Santa Catarina, Florianopolis, Brazil 🇧🇷 - [pretest] 
- Fuzhou University, Fuzhou, China 🇨🇳 - [pretest]
- Hangzhou Dianzi University, Hangzhou, China 🇨🇳 - [pretest] 
- Indian Institute of Science, Bangalore, India 🇮🇳 - [pretest] 
- Jinan University, Guangzhou, China 🇨🇳
- KU Leuven, Leuven, Belgium 🇧🇪 - [pretest] 
- Mondragon University, Hernani, Spain 🇪🇸 - [pretest] 
- Nanjing University of Posts and Telecom., Nanjing, China 🇨🇳 - [pretest] 
- Nanyang Technological University, Singapore 🇸🇬
- National Taipei University of Technology, Taipei, Taiwan 🇹🇼
- Northeastern University, Boston MA, USA 🇺🇸 - [pretest] 
- Paderborn University, Paderborn, Germany 🇩🇪 - [pretest] 
- Politecnico di Torino, Torino, Italy 🇮🇹 - [pretest] 
- Princeton University, Princeton NJ, USA 🇺🇸 (not competing)
- Purdue University, West Lafayette IN, USA 🇺🇸 - [pretest] 
- Seoul National University, Seoul, South Korea 🇰🇷
- Silicon Austria Labs, Graz, Austria 🇦🇹 - [pretest] 
- Southeast University Team 1, Nanjing, China 🇨🇳 - [pretest] 
- Southeast University Team 2, Nanjing, China 🇨🇳 - [pretest]
- Tribhuvan University, Lalitpur, Nepal 🇳🇵 - [pretest] 
- Tsinghua University, Beijing, China 🇨🇳 - [pretest] 
- TU Delft, Delft, Netherland 🇳🇱 - [pretest] 
- University of Bristol, Bristol, UK 🇬🇧 - [pretest] 
- University of Colorado Boulder, Boulder CO, USA 🇺🇸 - [pretest] 
- University of Kassel, Kassel, Germany 🇩🇪
- University of Manchester, Manchester, UK 🇬🇧
- University of Nottingham, Nottingham, UK 🇬🇧 - [pretest] 
- University of Sydney, Sydney, Australia 🇦🇺 - [pretest] 
- University of Tennessee, Knoxville, USA 🇺🇸 - [pretest] 
- University of Twente Team 1, Enschede, Netherland 🇳🇱 - [pretest] 
- University of Twente Team 2, Enschede, Netherland 🇳🇱
- University of Wisconsin-Madison, Madison MI, USA 🇺🇸
- Universidad Politécnica de Madrid, Madrid, Spain 🇪🇸
- Xi'an Jiaotong University, Xi'an, China 🇨🇳 - [pretest] 
- Zhejiang University, Hangzhou, China 🇨🇳 - [pretest] 
- Zhejiang University-UIUC, Hangzhou, China 🇨🇳 - [pretest] 

## Related Websites

- [MagNet Challenge Homepage](https://minjiechen.github.io/magnetchallenge/)
- [MagNet Challenge GitHub](https://github.com/minjiechen/magnetchallenge)
- [MagNet-AI Platform](https://mag-net.princeton.edu/)
- [MagNet-AI GitHub](https://github.com/PrincetonUniversity/Magnet)
- [Princeton Power Electronics Research Lab](https://www.princeton.edu/~minjie/magnet.html)
- [Dartmouth PMIC](https://pmic.engineering.dartmouth.edu/)
- [ETHz PES](https://pes.ee.ethz.ch/)

## MagNet Project Reference Papers

- D. Serrano et al., "Why MagNet: Quantifying the Complexity of Modeling Power Magnetic Material Characteristics," in IEEE Transactions on Power Electronics, doi: 10.1109/TPEL.2023.3291084. [Paper](https://ieeexplore.ieee.org/document/10169101)
- H. Li et al., "How MagNet: Machine Learning Framework for Modeling Power Magnetic Material Characteristics," in IEEE Transactions on Power Electronics, doi: 10.1109/TPEL.2023.3309232. [Paper](https://ieeexplore.ieee.org/document/10232863)
- H. Li, D. Serrano, S. Wang and M. Chen, "MagNet-AI: Neural Network as Datasheet for Magnetics Modeling and Material Recommendation," in IEEE Transactions on Power Electronics, doi: 10.1109/TPEL.2023.3309233. [Paper](https://ieeexplore.ieee.org/document/10232911)

## Organizers
<img src="img/magnetteam.jpg" width="800">

## Sponsors
<img src="img/sponsor.jpg" width="800">

