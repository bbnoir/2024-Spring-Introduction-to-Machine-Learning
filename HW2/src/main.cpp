#include "include.hpp"
#include "plot.hpp"
#include "dataloader.hpp"
#include "gen_model.hpp"
#include "dis_model.hpp"

int main()
{
    DataLoader* dl_train = new DataLoader("HW2_training.csv");
    DataLoader* dl_test = new DataLoader("HW2_testing.csv");

    std::cout << "=== Part 1: 4 classes ===\n\n";

    std::cout << "==== Generative Model ====\n\n";
    GenModel* gen_model = new GenModel(2, 4);
    std::cout << "---- Training ----\n";
    gen_model->Train(dl_train);
    gen_model->Test(dl_train);
    std::cout << "\n---- Testing ----\n";
    gen_model->Test(dl_test);

    DataLoader* dl_plot = GenPlotData(1000);
    vector_t gen_plot_y = gen_model->TestQuiet(dl_plot);
    WritePlotData(&gen_plot_y, dl_plot, "results/gen_model.csv");

    std::cout << "\n==== Discriminative Model ====\n";
    DisModel* dis_model = new DisModel(2, 4);
    std::cout << "\n---- Training ----\n";
    dis_model->Train(dl_train);
    dis_model->Test(dl_train);
    std::cout << "\n---- Testing ----\n";
    dis_model->Test(dl_test);

    vector_t dis_plot_y = dis_model->TestQuiet(dl_plot);
    WritePlotData(&dis_plot_y, dl_plot, "results/dis_model.csv");

    std::cout << "\n=== Part 2: 3 classes ===\n\n";

    // turn class 3 into class 0
    dl_train->TransClass3To0();
    dl_test->TransClass3To0();

    std::cout << "==== Generative Model ====\n\n";
    delete gen_model;
    gen_model = new GenModel(2, 3);
    std::cout << "---- Training ----\n";
    gen_model->Train(dl_train);
    gen_model->Test(dl_train);
    std::cout << "\n---- Testing ----\n";
    gen_model->Test(dl_test);

    vector_t gen_plot_y2 = gen_model->TestQuiet(dl_plot);
    WritePlotData(&gen_plot_y2, dl_plot, "results/gen_model2.csv");

    std::cout << "\n==== Discriminative Model ====\n";
    delete dis_model;
    dis_model = new DisModel(2, 3);
    std::cout << "\n---- Training ----\n";
    dis_model->Train(dl_train);
    dis_model->Test(dl_train);
    std::cout << "\n---- Testing ----\n";
    dis_model->Test(dl_test);

    vector_t dis_plot_y2 = dis_model->TestQuiet(dl_plot);
    WritePlotData(&dis_plot_y2, dl_plot, "results/dis_model2.csv");

    return 0;
}
