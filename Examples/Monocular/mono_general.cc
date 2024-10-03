/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <boost/filesystem.hpp> 

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strImagePath, vector<string> &vstrImages, vector<double> &vTimeStamps);
// void LoadImages(const string &strImagePath, const string &strPathTimes,
//                 vector<string> &vstrImages, vector<double> &vTimeStamps);

int main(int argc, char **argv)
{
    if(argc < 4)
    {
        cerr << endl << "Usage: ./mono_euroc path_to_vocabulary path_to_settings path_to_sequence_folder_1 (path_to_image_folder_2 ... path_to_image_folder_N) (trajectory_file_name)" << endl;
        return 1;
    }

    const int num_seq = (argc - 3) / 1;
    bool bFileName = (((argc - 3) % 1) == 1);
    string file_name;
    if (bFileName)
    {
        file_name = string(argv[argc - 1]);
        cout << "file name: " << file_name << endl;
    }

    vector< vector<string> > vstrImageFilenames;
    vector< vector<double> > vTimestampsCam;
    vector<int> nImages;

    vstrImageFilenames.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    nImages.resize(num_seq);

    int tot_images = 0;
    for (int seq = 0; seq < num_seq; seq++)
    {
        cout << "Loading images for sequence " << seq << "...";
        LoadImages(string(argv[seq + 3]), vstrImageFilenames[seq], vTimestampsCam[seq]);
        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageFilenames[seq].size();
        tot_images += nImages[seq];
    }

    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    cout.precision(17);

    int fps = 20;
    float dT = 1.f / fps;
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    double t_resize = 0.f;
    double t_track = 0.f;

    for (int seq = 0; seq < num_seq; seq++)
    {
        cv::Mat im;
        int proccIm = 0;
        for (int ni = 0; ni < nImages[seq]; ni++, proccIm++)
        {
            im = cv::imread(vstrImageFilenames[seq][ni], cv::IMREAD_UNCHANGED);
            double tframe = vTimestampsCam[seq][ni];

            if (im.empty())
            {
                cerr << endl << "Failed to load image at: " << vstrImageFilenames[seq][ni] << endl;
                return 1;
            }

            if (imageScale != 1.f)
            {
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
    #else
                std::chrono::monotonic_clock::time_point t_Start_Resize = std::chrono::monotonic_clock::now();
    #endif
#endif
                int width = im.cols * imageScale;
                int height = im.rows * imageScale;
                cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
                std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
    #else
                std::chrono::monotonic_clock::time_point t_End_Resize = std::chrono::monotonic_clock::now();
    #endif
                t_resize = std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t_End_Resize - t_Start_Resize).count();
                SLAM.InsertResizeTime(t_resize);
#endif
            }

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #else
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #endif

            Sophus::SE3f Tcw = SLAM.TrackMonocular(im, tframe);
            // if (Tcw.translation().norm() > 0) {
            //     cout << "Tcw translation: " << Tcw.translation().transpose() << endl;
            //     cout << "Tcw rotation: " << Tcw.rotationMatrix() << endl;
            // } else {
            //     // cout << "Tracking failed" << endl;
            // }

    #ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    #else
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    #endif

#ifdef REGISTER_TIMES
            t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double, std::milli> >(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif

            double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            vTimesTrack[ni] = ttrack;

            double T = (ni < nImages[seq] - 1) ? vTimestampsCam[seq][ni + 1] - tframe : (ni > 0) ? tframe - vTimestampsCam[seq][ni - 1] : 0;

            if (ttrack < T)
            {
                usleep((T - ttrack) * 1e6);
            }
        }

        if (seq < num_seq - 1)
        {
            string kf_file_submap = "./SubMaps/kf_SubMap_" + std::to_string(seq) + ".txt";
            string f_file_submap = "./SubMaps/f_SubMap_" + std::to_string(seq) + ".txt";
            SLAM.SaveTrajectoryEuRoC(f_file_submap);
            SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file_submap);

            cout << "Changing the dataset" << endl;
            SLAM.ChangeDataset();
        }
    }

    SLAM.Shutdown();

    if (bFileName)
    {
        const string kf_file = "kf_" + string(argv[argc - 1]) + ".txt";
        const string f_file = "f_" + string(argv[argc - 1]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else
    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }

    return 0;
}
// void LoadImages(const string &strImagePath, const string &strPathTimes,
//                 vector<string> &vstrImages, vector<double> &vTimeStamps)
// {
//     ifstream fTimes;
//     fTimes.open(strPathTimes.c_str());
//     vTimeStamps.reserve(5000);
//     vstrImages.reserve(5000);
//     while(!fTimes.eof())
//     {
//         string s;
//         getline(fTimes,s);
//         if(!s.empty())
//         {
//             stringstream ss;
//             ss << s;
//             vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
//             double t;
//             ss >> t;
//             vTimeStamps.push_back(t*1e-9);

//         }
//     }
// }


void LoadImages(const std::string &strImagePath, std::vector<std::string> &vstrImages, std::vector<double> &vTimeStamps)
{
    namespace fs = boost::filesystem;

    // Clear vectors
    vstrImages.clear();
    vTimeStamps.clear();

    // Iterate through directory and collect image files
    for (const auto &entry : fs::directory_iterator(strImagePath))
    {
        if (fs::is_regular_file(entry))
        {
            std::string extension = entry.path().extension().string();
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg")
            {
                std::string filename = entry.path().stem().string();
                vstrImages.push_back(entry.path().string());
                vTimeStamps.push_back(std::stod(filename) * 1e-9); // Convert to seconds
            }
        }
    }

    // Sort images and timestamps based on timestamps
    std::vector<std::pair<double, std::string>> timestampedImages;
    for (size_t i = 0; i < vstrImages.size(); ++i)
    {
        timestampedImages.emplace_back(vTimeStamps[i], vstrImages[i]);
    }

    std::sort(timestampedImages.begin(), timestampedImages.end());

    // Refill vectors with sorted data
    vstrImages.clear();
    vTimeStamps.clear();
    for (const auto &pair : timestampedImages)
    {
        vTimeStamps.push_back(pair.first);
        vstrImages.push_back(pair.second);
    }
}
