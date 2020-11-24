#include <iostream>
#include <vector>
using std::cout;
using std::endl;
using std::vector;
using std::string;

#include <glog/logging.h>

#include <opencv2/opencv.hpp>
#include "descriptor_opencvm.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

int main(int argc, char** argv) {
    //读入图像
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    FLAGS_logtostderr=true;

    double fx=481.2;
    double fy=480;
    double cx=319.5;
    double cy=239.5;
    //读入图像
    cv::Mat color1 = cv::imread("../0.png",-1);     //先读入一帧RGB和depth
    cv::Mat color2 = cv::imread("../50.png",-1);
    if ( color1.data==nullptr || color2.empty()){
        cout<<"wrong input rgb image"<<endl;
    }
    //初始化
    cv::Ptr<cvm::line_descriptor::LSDDetector> lsd = cvm::line_descriptor::LSDDetector::createLSDDetector();
    cv::Ptr<cvm::line_descriptor::BinaryDescriptor> lbd = cvm::line_descriptor::BinaryDescriptor::createBinaryDescriptor();

    //提取图1的线特征
    std::vector<cvm::line_descriptor::KeyLine> keylines1;
    lsd->detect(color1, keylines1, 1.2, 1);  //提取线特征并计算描述子
    LOG(INFO)<<"Extract line1 num = "<<keylines1.size();
    cv::Mat ldesc1;
    lbd->compute(color1, keylines1, ldesc1);     //计算特征线段的描述子
    cv::Mat outlinesimg1;
    cvm::line_descriptor::drawKeylines(color1,keylines1,outlinesimg1);
    cv::imwrite("../colorline1.png",outlinesimg1);

    //提取图2的线特征
    std::vector<cvm::line_descriptor::KeyLine> keylines2;
    lsd->detect(color2, keylines2, 1.2, 1);  //提取线特征并计算描述子
    LOG(INFO)<<"Extract line2 num = "<<keylines2.size();
    cv::Mat ldesc2;
    lbd->compute(color2, keylines2, ldesc2);     //计算特征线段的描述子
    cv::Mat outlinesimg2;
    cvm::line_descriptor::drawKeylines(color2,keylines2,outlinesimg2);
    cv::imwrite("../colorline2.png",outlinesimg2);
    //匹配
    cv::Ptr<cvm::line_descriptor::BinaryDescriptorMatcher> bdm=cvm::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    std::vector<cv::DMatch> matches;
    bdm->match(ldesc1,ldesc2,matches);
    cv::Mat out_match_img;
    LOG(INFO)<<"match num = "<<matches.size();

    const std::vector<char> matchesMask(matches.size(),1);
    cvm::line_descriptor::drawLineMatches(color1,keylines1,color2,keylines2,matches,out_match_img,
            cv::Scalar::all(-1),cv::Scalar::all(-1),matchesMask,cvm::line_descriptor::DrawLinesMatchesFlags::DEFAULT);
    cv::imwrite("../match_1_2.png",out_match_img);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
