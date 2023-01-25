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

#include "agentRotatingWallVelocityFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::agentRotatingWallVelocityFvPatchVectorField::
    agentRotatingWallVelocityFvPatchVectorField(
        const fvPatch &p,
        const DimensionedField<vector, volMesh> &iF)
    : fixedValueFvPatchField<vector>(p, iF),
      origin_(),
      axis_(Zero),
      probes_(initializeProbes()),
      control_(initializeControl())
{}

Foam::agentRotatingWallVelocityFvPatchVectorField::
    agentRotatingWallVelocityFvPatchVectorField(
        const fvPatch &p,
        const DimensionedField<vector, volMesh> &iF,
        const dictionary &dict)
    : fixedValueFvPatchField<vector>(p, iF, dict, false),
      origin_(dict.get<vector>("origin")),
      axis_(dict.get<vector>("axis")),
      train_(dict.get<bool>("train")),
      policy_name_(dict.get<word>("policy")),
      policy_(torch::jit::load(policy_name_)),
      abs_omega_max_(dict.get<scalar>("absOmegaMax")),
      seed_(dict.get<int>("seed")),
      probes_name_(dict.get<word>("probesDict")),
      gen_(seed_),
      omega_(0.0),
      omega_old_(0.0),
      control_time_(0.0),
      update_omega_(true),
      probes_(initializeProbes()),
      control_(initializeControl())
{
      updateCoeffs();
}


Foam::agentRotatingWallVelocityFvPatchVectorField::
    agentRotatingWallVelocityFvPatchVectorField(
        const agentRotatingWallVelocityFvPatchVectorField &ptf,
        const fvPatch &p,
        const DimensionedField<vector, volMesh> &iF,
        const fvPatchFieldMapper &mapper)
    : fixedValueFvPatchField<vector>(ptf, p, iF, mapper),
      origin_(ptf.origin_),
      axis_(ptf.axis_),
      train_(ptf.train_),
      policy_name_(ptf.policy_name_),
      policy_(ptf.policy_),
      abs_omega_max_(ptf.abs_omega_max_),
      seed_(ptf.seed_),
      probes_name_(ptf.probes_name_),
      gen_(ptf.gen_),
      omega_(ptf.omega_),
      omega_old_(ptf.omega_old_),
      control_time_(ptf.control_time_),
      update_omega_(ptf.update_omega_),
      probes_(initializeProbes()),
      control_(initializeControl())
{}

Foam::agentRotatingWallVelocityFvPatchVectorField::
    agentRotatingWallVelocityFvPatchVectorField(
        const agentRotatingWallVelocityFvPatchVectorField &rwvpvf)
    : fixedValueFvPatchField<vector>(rwvpvf),
      origin_(rwvpvf.origin_),
      axis_(rwvpvf.axis_),
      train_(rwvpvf.train_),
      policy_name_(rwvpvf.policy_name_),
      policy_(rwvpvf.policy_),
      abs_omega_max_(rwvpvf.abs_omega_max_),
      seed_(rwvpvf.seed_),
      probes_name_(rwvpvf.probes_name_),
      gen_(rwvpvf.gen_),
      omega_(rwvpvf.omega_),
      omega_old_(rwvpvf.omega_old_),
      control_time_(rwvpvf.control_time_),
      update_omega_(rwvpvf.update_omega_),
      probes_(initializeProbes()),
      control_(initializeControl())
{}

Foam::agentRotatingWallVelocityFvPatchVectorField::
    agentRotatingWallVelocityFvPatchVectorField(
        const agentRotatingWallVelocityFvPatchVectorField &rwvpvf,
        const DimensionedField<vector, volMesh> &iF)
    : fixedValueFvPatchField<vector>(rwvpvf, iF),
      origin_(rwvpvf.origin_),
      axis_(rwvpvf.axis_),
      train_(rwvpvf.train_),
      policy_name_(rwvpvf.policy_name_),
      policy_(rwvpvf.policy_),
      abs_omega_max_(rwvpvf.abs_omega_max_),
      seed_(rwvpvf.seed_),
      probes_name_(rwvpvf.probes_name_),
      gen_(rwvpvf.gen_),
      omega_(rwvpvf.omega_),
      omega_old_(rwvpvf.omega_old_),
      control_time_(rwvpvf.control_time_),
      update_omega_(rwvpvf.update_omega_),
      probes_(initializeProbes()),
      control_(initializeControl())
{}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::agentRotatingWallVelocityFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    // update angular velocity
    const scalar t = this->db().time().timeOutputValue();
    const scalar dt = this->db().time().deltaTValue();
    const scalar timeIndex = this->db().time().timeIndex();
    const scalar startTimeIndex = this->db().time().startTimeIndex();
    bool timeToControl = control_.execute() &&
                         t >= start_time_ - 0.5*dt &&
                         timeIndex != startTimeIndex;

    if (timeToControl && update_omega_)
    {
        omega_old_ = omega_;
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
            scalar alpha = dist_parameters[0][0].item<double>();
            scalar beta = dist_parameters[0][1].item<double>();
            std::gamma_distribution<double> distribution_1(alpha, 1.0);
            std::gamma_distribution<double> distribution_2(beta, 1.0);
            scalar omega_pre_scale;
            if (train_)
            {
                // sample from Beta distribution during training
                double number_1 = distribution_1(gen_);
                double number_2 = distribution_2(gen_);
                omega_pre_scale = number_1 / (number_1 + number_2);
            }
            else
            {
                // use expected (mean) angular velocity
                omega_pre_scale = alpha / (alpha + beta);
            }
            // rescale to actionspace
            omega_ = (omega_pre_scale - 0.5) * 2 * abs_omega_max_;
            // save trajectory
            saveTrajectory(alpha, beta);
            Info << "New omega: " << omega_ << "; old value: " << omega_old_ << "\n";
        }
        Pstream::scatter(omega_);

        // avoid update of angular velocity during p-U coupling
        update_omega_ = false;
    }

    // activate update of angular velocity after p-U coupling
    if (!timeToControl && !update_omega_)
    {
        update_omega_ = true;
    }

    // update angular velocity by linear transition from old to new value
    
    scalar d_omega = (omega_ - omega_old_) / dt_control_* (t - control_time_);
    scalar omega = omega_old_ + d_omega;

    // Calculate the rotating wall velocity from the specification of the motion
    const vectorField Up(
        (-omega) * ((patch().Cf() - origin_) ^ (axis_ / mag(axis_))));

    // Remove the component of Up normal to the wall
    // just in case it is not exactly circular
    const vectorField n(patch().nf());
    vectorField::operator=(Up - n * (n & Up));

    fixedValueFvPatchVectorField::updateCoeffs();
}

void Foam::agentRotatingWallVelocityFvPatchVectorField::write(Ostream &os) const
{
    fvPatchVectorField::write(os);
    os.writeEntry("origin", origin_);
    os.writeEntry("axis", axis_);
    os.writeEntry("policy", policy_name_);
    os.writeEntry("train", train_);
    os.writeEntry("absOmegaMax", abs_omega_max_);
    os.writeEntry("seed", seed_);
    os.writeEntry("probesDict", probes_name_);
}

void Foam::agentRotatingWallVelocityFvPatchVectorField::saveTrajectory(scalar alpha, scalar beta) const
{
    std::ifstream file("trajectory.csv");
    std::fstream trajectory("trajectory.csv", std::ios::app | std::ios::binary);
    const scalar t = this->db().time().timeOutputValue();

    if(!file.good())
    {
        // write header
        trajectory << "t, omega, alpha, beta";
    }

    trajectory << std::setprecision(15)
               << "\n"
               << t << ", "
               << omega_ << ", "
               << alpha << ", "
               << beta;
}

const Foam::dictionary& Foam::agentRotatingWallVelocityFvPatchVectorField::getProbesDict()
{
    const dictionary& funcDict = this->db().time().controlDict().subDict("functions");
    if (!funcDict.found(probes_name_))
    {
        FatalError << "probesDict" << probes_name_ << " not found\n" << exit(FatalError);
        
    }
    return funcDict.subDict(probes_name_);
}

Foam::probes Foam::agentRotatingWallVelocityFvPatchVectorField::initializeProbes()
{
    const dictionary& probesDict = getProbesDict();
    return Foam::probes("probes", this->db().time(), probesDict, false, true);
}

Foam::timeControl Foam::agentRotatingWallVelocityFvPatchVectorField::initializeControl()
{
    const dictionary& probesDict = getProbesDict();
    start_time_ = probesDict.getOrDefault<scalar>("timeStart", 0.0);
    dt_control_ = probesDict.getOrDefault<scalar>("executeInterval", 1.0);
    return Foam::timeControl(this->db().time(), probesDict, "execute");
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField(
        fvPatchVectorField,
        agentRotatingWallVelocityFvPatchVectorField);
}

// ************************************************************************* //
