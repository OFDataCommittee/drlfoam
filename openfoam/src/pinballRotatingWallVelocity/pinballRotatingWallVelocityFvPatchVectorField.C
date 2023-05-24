/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/
#include "pinballRotatingWallVelocityFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::pinballRotatingWallVelocityFvPatchVectorField::
    pinballRotatingWallVelocityFvPatchVectorField(
        const fvPatch &p,
        const DimensionedField<vector, volMesh> &iF)
    : fixedValueFvPatchField<vector>(p, iF),
      origin_a_(),
      origin_b_(),
      origin_c_(),
      axis_(Zero),
      probes_(initializeProbes()),
      control_(initializeControl())
{
    initializeCylinderSegmentation();
}

Foam::pinballRotatingWallVelocityFvPatchVectorField::
    pinballRotatingWallVelocityFvPatchVectorField(
        const fvPatch &p,
        const DimensionedField<vector, volMesh> &iF,
        const dictionary &dict)
    : fixedValueFvPatchField<vector>(p, iF, dict, false),
      origin_a_(dict.get<vector>("origin_a")),
      origin_b_(dict.get<vector>("origin_b")),
      origin_c_(dict.get<vector>("origin_c")),
      axis_(dict.get<vector>("axis")),
      train_(dict.get<bool>("train")),
      policy_name_(dict.get<word>("policy")),
      policy_(torch::jit::load(policy_name_)),
      abs_omega_max_(dict.get<scalar>("absOmegaMax")),
      seed_(dict.get<int>("seed")),
      probes_name_(dict.get<word>("probesDict")),
      gen_(seed_),
      omega_a_(0.0),
      omega_old_a_(0),
      omega_b_(0.0),
      omega_old_b_(0),
      omega_c_(0.0),
      omega_old_c_(0),
      control_time_(0.0),
      update_omega_(true),
      probes_(initializeProbes()),
      control_(initializeControl())
{
    initializeCylinderSegmentation();
    updateCoeffs();
}


Foam::pinballRotatingWallVelocityFvPatchVectorField::
    pinballRotatingWallVelocityFvPatchVectorField(
        const pinballRotatingWallVelocityFvPatchVectorField &ptf,
        const fvPatch &p,
        const DimensionedField<vector, volMesh> &iF,
        const fvPatchFieldMapper &mapper)
    : fixedValueFvPatchField<vector>(ptf, p, iF, mapper),
      origin_a_(ptf.origin_a_),
      origin_b_(ptf.origin_b_),
      origin_c_(ptf.origin_c_),
      axis_(ptf.axis_),
      train_(ptf.train_),
      policy_name_(ptf.policy_name_),
      policy_(ptf.policy_),
      abs_omega_max_(ptf.abs_omega_max_),
      seed_(ptf.seed_),
      probes_name_(ptf.probes_name_),
      gen_(ptf.gen_),
      omega_a_(ptf.omega_a_),
      omega_old_a_(ptf.omega_old_a_),
      omega_b_(ptf.omega_b_),
      omega_old_b_(ptf.omega_old_b_),
      omega_c_(ptf.omega_c_),
      omega_old_c_(ptf.omega_old_c_),
      control_time_(ptf.control_time_),
      update_omega_(ptf.update_omega_),
      probes_(initializeProbes()),
      control_(initializeControl())
{
    initializeCylinderSegmentation();
}

Foam::pinballRotatingWallVelocityFvPatchVectorField::
    pinballRotatingWallVelocityFvPatchVectorField(
        const pinballRotatingWallVelocityFvPatchVectorField &rwvpvf)
    : fixedValueFvPatchField<vector>(rwvpvf),
      origin_a_(rwvpvf.origin_a_),
      origin_b_(rwvpvf.origin_b_),
      origin_c_(rwvpvf.origin_c_),
      axis_(rwvpvf.axis_),
      train_(rwvpvf.train_),
      policy_name_(rwvpvf.policy_name_),
      policy_(rwvpvf.policy_),
      abs_omega_max_(rwvpvf.abs_omega_max_),
      seed_(rwvpvf.seed_),
      probes_name_(rwvpvf.probes_name_),
      gen_(rwvpvf.gen_),
      omega_a_(rwvpvf.omega_a_),
      omega_old_a_(rwvpvf.omega_old_a_),
      omega_b_(rwvpvf.omega_b_),
      omega_old_b_(rwvpvf.omega_old_b_),
      omega_c_(rwvpvf.omega_c_),
      omega_old_c_(rwvpvf.omega_old_c_),
      control_time_(rwvpvf.control_time_),
      update_omega_(rwvpvf.update_omega_),
      probes_(initializeProbes()),
      control_(initializeControl())
{
    initializeCylinderSegmentation();
}

Foam::pinballRotatingWallVelocityFvPatchVectorField::
    pinballRotatingWallVelocityFvPatchVectorField(
        const pinballRotatingWallVelocityFvPatchVectorField &rwvpvf,
        const DimensionedField<vector, volMesh> &iF)
    : fixedValueFvPatchField<vector>(rwvpvf, iF),
      origin_a_(rwvpvf.origin_a_),
      origin_b_(rwvpvf.origin_b_),
      origin_c_(rwvpvf.origin_c_),
      axis_(rwvpvf.axis_),
      train_(rwvpvf.train_),
      policy_name_(rwvpvf.policy_name_),
      policy_(rwvpvf.policy_),
      abs_omega_max_(rwvpvf.abs_omega_max_),
      seed_(rwvpvf.seed_),
      probes_name_(rwvpvf.probes_name_),
      gen_(rwvpvf.gen_),
      omega_a_(rwvpvf.omega_a_),
      omega_old_a_(rwvpvf.omega_old_a_),
      omega_b_(rwvpvf.omega_b_),
      omega_old_b_(rwvpvf.omega_old_b_),
      omega_c_(rwvpvf.omega_c_),
      omega_old_c_(rwvpvf.omega_old_c_),
      control_time_(rwvpvf.control_time_),
      update_omega_(rwvpvf.update_omega_),
      probes_(initializeProbes()),
      control_(initializeControl())
{
    initializeCylinderSegmentation();
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::pinballRotatingWallVelocityFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    // update angular velocities
    const scalar t = this->db().time().timeOutputValue();
    const scalar dt = this->db().time().deltaTValue();
    const scalar timeIndex = this->db().time().timeIndex();
    const scalar startTimeIndex = this->db().time().startTimeIndex();
    bool timeToControl = control_.execute() &&
                         t >= start_time_ - 0.5*dt &&
                         timeIndex != startTimeIndex;

    if (timeToControl && update_omega_)
    {
        omega_old_a_ = omega_a_;
        omega_old_b_ = omega_b_;
        omega_old_c_ = omega_c_;
        control_time_ = t;

        const volScalarField& p = this->db().lookupObject<volScalarField>("p"); 
        scalarField p_sample = probes_.sample(p);

        if (Pstream::master()) // evaluate policy only on the master
        {
            torch::Tensor features = torch::from_blob(
                p_sample.data(), {1, p_sample.size()}, torch::TensorOptions().dtype(torch::kFloat64)
            );
            std::vector<torch::jit::IValue> policyFeatures{features};
            torch::Tensor dist_parameters = policy_.forward(policyFeatures).toTensor();
            scalar alpha_a = dist_parameters[0][0].item<double>();
            scalar alpha_b = dist_parameters[0][1].item<double>();
            scalar alpha_c = dist_parameters[0][2].item<double>();
            scalar beta_a = dist_parameters[0][3].item<double>();
            scalar beta_b = dist_parameters[0][4].item<double>();
            scalar beta_c = dist_parameters[0][5].item<double>();
            std::gamma_distribution<double> distribution_1_a(alpha_a, 1.0);
            std::gamma_distribution<double> distribution_2_a(alpha_b, 1.0);
            std::gamma_distribution<double> distribution_3_a(alpha_c, 1.0);
            std::gamma_distribution<double> distribution_1_b(beta_a, 1.0);
            std::gamma_distribution<double> distribution_2_b(beta_b, 1.0);
            std::gamma_distribution<double> distribution_3_b(beta_c, 1.0);
            scalar omega_pre_scale_a;
            scalar omega_pre_scale_b;
            scalar omega_pre_scale_c;
            if (train_)
            {
                // sample from Beta distribution during training
                double number_1_a = distribution_1_a(gen_);
                double number_2_a = distribution_2_a(gen_);
                double number_3_a = distribution_3_a(gen_);
                double number_1_b = distribution_1_b(gen_);
                double number_2_b = distribution_2_b(gen_);
                double number_3_b = distribution_3_b(gen_);
                omega_pre_scale_a = number_1_a / (number_1_a + number_1_b);
                omega_pre_scale_b = number_2_a / (number_2_a + number_2_b);
                omega_pre_scale_c = number_3_a / (number_3_a + number_3_b);
            }
            else
            {
                // use expected (mean) angular velocity
                omega_pre_scale_a = alpha_a / (alpha_a + beta_a);
                omega_pre_scale_b = alpha_b / (alpha_b + beta_b);
                omega_pre_scale_c = alpha_c / (alpha_c + beta_c);
            }
            // rescale to actionspace
            omega_a_ = (omega_pre_scale_a - 0.5) * 2 * abs_omega_max_;
            omega_b_ = (omega_pre_scale_b - 0.5) * 2 * abs_omega_max_;
            omega_c_ = (omega_pre_scale_c - 0.5) * 2 * abs_omega_max_;
            // save trajectory
            saveTrajectory(alpha_a, beta_a, alpha_b, beta_b, alpha_c, beta_c);
            Info << "New omega_a: " << omega_a_ << "; old value: " << omega_old_a_ << "\n";
            Info << "New omega_b: " << omega_b_ << "; old value: " << omega_old_b_ << "\n";
            Info << "New omega_c: " << omega_c_ << "; old value: " << omega_old_c_ << "\n";
        }
        Pstream::scatter(omega_a_);
        Pstream::scatter(omega_b_);
        Pstream::scatter(omega_c_);
        // avoid update of angular velocity during p-U coupling
        update_omega_ = false;
    }

    // activate update of angular velocity after p-U coupling
    if (!timeToControl && !update_omega_)
    {
        update_omega_ = true;
    }

    // update angular velocity by linear transition from old to new value
    scalar d_omega_a = (omega_a_ - omega_old_a_) / dt_control_ * (t - control_time_);
    scalar omega_a = omega_old_a_ + d_omega_a;
    scalar d_omega_b = (omega_b_ - omega_old_b_) / dt_control_ * (t - control_time_);
    scalar omega_b = omega_old_b_ + d_omega_b;
    scalar d_omega_c = (omega_c_ - omega_old_c_) / dt_control_ * (t - control_time_);
    scalar omega_c = omega_old_c_ + d_omega_c;

    const int patch_size = (patch().Cf()).size();
    vectorField Up(patch_size);
    forAll(faces_a_, faceI)
    {
        label faceAI = faces_a_[faceI];
        Up[faceAI] = (-omega_a) * ((centers_a_[faceI] - origin_a_) ^ (axis_ / mag(axis_)));
        Up[faceAI] -= normals_a_[faceI] * (normals_a_[faceI] & Up[faceAI]);
    }
    forAll(faces_b_, faceI)
    {
        label faceBI = faces_b_[faceI];
        Up[faceBI] = (-omega_b) * ((centers_b_[faceI] - origin_b_) ^ (axis_ / mag(axis_)));
        Up[faceBI] -= normals_b_[faceI] * (normals_b_[faceI] & Up[faceBI]);
    }
    forAll(faces_c_, faceI)
    {
        label faceCI = faces_c_[faceI];
        Up[faceCI] = (-omega_c) * ((centers_c_[faceI] - origin_c_) ^ (axis_ / mag(axis_)));
        Up[faceCI] -= normals_c_[faceI] * (normals_c_[faceI] & Up[faceCI]);
    }

    vectorField::operator=(Up);
    fixedValueFvPatchVectorField::updateCoeffs();
}

void Foam::pinballRotatingWallVelocityFvPatchVectorField::write(Ostream &os) const
{
    fvPatchVectorField::write(os);
    os.writeEntry("origin_a", origin_a_);
    os.writeEntry("origin_b", origin_b_);
    os.writeEntry("origin_c", origin_c_);
    os.writeEntry("axis", axis_);
    os.writeEntry("policy", policy_name_);
    os.writeEntry("train", train_);
    os.writeEntry("absOmegaMax", abs_omega_max_);
    os.writeEntry("seed", seed_);
    os.writeEntry("probesDict", probes_name_);
}

void Foam::pinballRotatingWallVelocityFvPatchVectorField::saveTrajectory(scalar alpha_a, scalar beta_a, scalar alpha_b, scalar beta_b, scalar alpha_c, scalar beta_c) const
{
    std::ifstream file("trajectory.csv");
    std::fstream trajectory("trajectory.csv", std::ios::app | std::ios::binary);
    const scalar t = this->db().time().timeOutputValue();

    if(!file.good())
    {
        // write header
        trajectory << "t, omega_a, alpha_a, beta_a, omega_b, alpha_b, beta_b, omega_c, alpha_c, beta_c";
    }

    trajectory << std::setprecision(15)
               << "\n"
               << t << ", "
               << omega_a_ << ", "
               << alpha_a << ", "
               << beta_a << ", "
               << omega_b_ << ", "
               << alpha_b << ", "
               << beta_b << ", "
               << omega_c_ << ", "
               << alpha_c << ", "
               << beta_c;
               

}

const Foam::dictionary& Foam::pinballRotatingWallVelocityFvPatchVectorField::getProbesDict()
{
    const dictionary& funcDict = this->db().time().controlDict().subDict("functions");
    if (!funcDict.found(probes_name_))
    {
        FatalError << "probesDict" << probes_name_ << " not found\n" << exit(FatalError);
        
    }
    return funcDict.subDict(probes_name_);
}

Foam::probes Foam::pinballRotatingWallVelocityFvPatchVectorField::initializeProbes()
{
    const dictionary& probesDict = getProbesDict();
    return Foam::probes("probes", this->db().time(), probesDict, false, true);
}

Foam::timeControl Foam::pinballRotatingWallVelocityFvPatchVectorField::initializeControl()
{
    const dictionary& probesDict = getProbesDict();
    start_time_ = probesDict.getOrDefault<scalar>("timeStart", 0.0);
    dt_control_ = probesDict.getOrDefault<scalar>("executeInterval", 1.0);
    return Foam::timeControl(this->db().time(), probesDict, "execute");
}

void Foam::pinballRotatingWallVelocityFvPatchVectorField::initializeCylinderSegmentation()
{
    const fvMesh& mesh(patch().boundaryMesh().mesh());
    // patch name and radius are currently hardcoded
    label patchID = mesh.boundaryMesh().findPatchID("cylinders");
    scalar radius = 0.5;
    const polyPatch& cPatch = mesh.boundaryMesh()[patchID];
    const surfaceScalarField& magSf = mesh.magSf();
    const surfaceVectorField& Cf = mesh.Cf();
    const surfaceVectorField& Sf = mesh.Sf();

    forAll(cPatch, faceI)
    {
        scalar x = Cf.boundaryField()[patchID][faceI].x();
        scalar y = Cf.boundaryField()[patchID][faceI].y();
    
        scalar dist_a = sqrt(pow(x - origin_a_[0], 2) + pow(y - origin_a_[1], 2));
        scalar dist_b = sqrt(pow(x - origin_b_[0], 2) + pow(y - origin_b_[1], 2));
        scalar dist_c = sqrt(pow(x - origin_c_[0], 2) + pow(y - origin_c_[1], 2));

        if (dist_a < 1.2*radius)
        {
            centers_a_.append(Cf.boundaryField()[patchID][faceI]);
            normals_a_.append(Sf.boundaryField()[patchID][faceI]/magSf.boundaryField()[patchID][faceI]);
            faces_a_.append(faceI);
        }

        if (dist_b < 1.2*radius)
        {
            centers_b_.append(Cf.boundaryField()[patchID][faceI]);
            normals_b_.append(Sf.boundaryField()[patchID][faceI]/magSf.boundaryField()[patchID][faceI]);
            faces_b_.append(faceI);
        }

        if (dist_c < 1.2*radius)
        {
            centers_c_.append(Cf.boundaryField()[patchID][faceI]);
            normals_c_.append(Sf.boundaryField()[patchID][faceI]/magSf.boundaryField()[patchID][faceI]);
            faces_c_.append(faceI);
        }
    }

}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField(
        fvPatchVectorField,
        pinballRotatingWallVelocityFvPatchVectorField);
}

// ************************************************************************* //
