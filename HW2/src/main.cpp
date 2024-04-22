#include "include.hpp"
#include "plot.hpp"
#include "dataloader.hpp"
#include "gen_model.hpp"
#include "dis_model.hpp"

int main()
{
    DataLoader* dl_train = new DataLoader("HW2_training.csv");
    DataLoader* dl_test = new DataLoader("HW2_testing.csv");

    // std::cout << "==== Generative Model ====" << std::endl;
    // GenModel* gen_model = new GenModel(2, 4);
    // std::cout << std::endl;
    // std::cout << "---- Training ----" << std::endl;
    // gen_model->Train(dl_train);
    // gen_model->Test(dl_train);
    // std::cout << std::endl;
    // std::cout << "---- Testing ----" << std::endl;
    // gen_model->Test(dl_test);

    // DataLoader* dl_plot = GenPlotData(1000);
    // vector_t plot_y = gen_model->Test(dl_plot);
    // WritePlotData(&plot_y, dl_plot, "results/gen_model.csv");

    std::cout << "==== Discriminative Model ====" << std::endl;
    DisModel* dis_model = new DisModel(2, 4);
    std::cout << std::endl;
    std::cout << "---- Training ----" << std::endl;
    dis_model->Train(dl_train);
    dis_model->Test(dl_train);
    std::cout << std::endl;
    std::cout << "---- Testing ----" << std::endl;
    dis_model->Test(dl_test);

    return 0;
}
