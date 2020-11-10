from syndalib import modelgen

"""
def test_haversine():
    assert modelgen.haversine(52.370216, 4.895168, 52.520008,
                              13.404954) == 945793.4375088713


def test_generate_circle():
    outliers_rate_range = [0.10]
    dirs = ['fakedata', '', 'test']
    modelgen.generate_data(num_samples=1,
                     num_points_per_sample=2,
                     outliers_rate_range=outliers_rate_range,
                     model='line',
                     noise_perc=0.01,
                     dest='matlab',
                     dirs=dirs)
"""


def test_generate_conics():
    outliers_perc_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    noise_perc_range = [0.01]
    dirs = ["no_variant_noise_0.01", "train", ""]
    modelgen.generate_data(
        num_samples=512,
        num_points_per_sample=256,
        outliers_perc_range=outliers_perc_range,
        model="conics",
        noise_perc_range=noise_perc_range,
        dest="matlab",
        dirs=dirs,
        number_of_models=2,
    )

    assert True
