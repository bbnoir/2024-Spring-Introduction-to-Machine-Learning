#include "include.hpp"
#include "dataloader.hpp"
#include "gen_model.hpp"
#include "dis_model.hpp"

int main()
{
    DataLoader* dl_train = new DataLoader("HW2_demo_training.csv");
    DataLoader* dl_test = new DataLoader("HW2_demo_testing.csv");

    std::cout << "==== Generative Model ====\n\n";
    GenModel* gen_model = new GenModel(2, 4);
    std::cout << "---- Training ----\n";
    gen_model->Train(dl_train);
    gen_model->Test(dl_train);
    std::cout << "\n---- Testing ----\n";
    gen_model->Test(dl_test);

    std::cout << "\n==== Discriminative Model ====\n";
    DisModel* dis_model = new DisModel(2, 4);
    std::cout << "\n---- Training ----\n";
    dis_model->Train(dl_train);
    dis_model->Test(dl_train);
    std::cout << "\n---- Testing ----\n";
    dis_model->Test(dl_test);

    return 0;
}
