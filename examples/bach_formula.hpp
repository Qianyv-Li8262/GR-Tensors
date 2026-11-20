#pragma once
#include "bach_metric_reference.hpp"

template <typename Bach, typename Weyl>
struct BachExpressions { Bach bach; Weyl weyl; };

template <DiffScheme Scheme = DiffScheme::Central2>
auto make_bach_expressions() {
    constexpr auto scheme = Scheme;
    auto g = make_field<&metric>();
    auto inverse = inverse_field(g);
    auto pg = partial_field<mu, scheme>(g);
    auto Gamma = scale_field<1, 2>(
        rename_field<up<a>, up<kappa>>(inverse) *
        (rename_field<dn<b>, dn<kappa>, dn<c>>(pg) +
         rename_field<dn<c>, dn<kappa>, dn<b>>(pg) -
         rename_field<dn<kappa>, dn<b>, dn<c>>(pg)));
    auto pGamma = partial_field<mu, scheme>(Gamma);

    // R^rho_{b c d}; the storage order is [rho, c, b, d].
    auto q1 = rename_field<up<rho>, dn<c>, dn<lambda>>(Gamma) *
              rename_field<up<lambda>, dn<b>, dn<d>>(Gamma);
    auto q2 = rename_field<up<rho>, dn<d>, dn<lambda>>(Gamma) *
              rename_field<up<lambda>, dn<b>, dn<c>>(Gamma);
    auto R = q1 - q2 + rename_field<dn<c>, up<rho>, dn<b>, dn<d>>(pGamma) -
                       rename_field<dn<d>, up<rho>, dn<b>, dn<c>>(pGamma);
    auto Ricci = trace_field<rho, c>(R); // [b, d], Ric_bd = R^rho_{b rho d}
    auto scalar = rename_field<up<b>, up<d>>(inverse) * Ricci;
    auto lower = rename_field<dn<a>, dn<rho>>(g) * R; // R_abcd, storage [a,c,b,d]

    auto ricci_terms =
        rename_field<dn<a>, dn<c>>(g) * rename_field<dn<b>, dn<d>>(Ricci) -
        rename_field<dn<a>, dn<d>>(g) * rename_field<dn<b>, dn<c>>(Ricci) -
        rename_field<dn<b>, dn<c>>(g) * rename_field<dn<a>, dn<d>>(Ricci) +
        rename_field<dn<b>, dn<d>>(g) * rename_field<dn<a>, dn<c>>(Ricci);
    auto metric_terms = rename_field<dn<a>, dn<c>>(g) * rename_field<dn<b>, dn<d>>(g) -
                        rename_field<dn<a>, dn<d>>(g) * rename_field<dn<b>, dn<c>>(g);
    auto Weyl = lower - scale_field<1, 2>(ricci_terms) + scale_field<1, 6>(scalar) * metric_terms;

    // T_abcd := C_acbd. This view follows the physical storage order of Weyl.
    auto T = rename_field<dn<a>, dn<b>, dn<c>, dn<d>>(Weyl);
    auto nabla_T = partial_field<mu, scheme>(T) -
        rename_field<up<lambda>, dn<mu>, dn<a>>(Gamma) *
            rename_field<dn<lambda>, dn<b>, dn<c>, dn<d>>(T) -
        rename_field<up<lambda>, dn<mu>, dn<b>>(Gamma) *
            rename_field<dn<a>, dn<lambda>, dn<c>, dn<d>>(T) -
        rename_field<up<lambda>, dn<mu>, dn<c>>(Gamma) *
            rename_field<dn<a>, dn<b>, dn<lambda>, dn<d>>(T) -
        rename_field<up<lambda>, dn<mu>, dn<d>>(Gamma) *
            rename_field<dn<a>, dn<b>, dn<c>, dn<lambda>>(T);
    auto D = rename_field<up<d>, up<mu>>(inverse) * nabla_T; // D_abc = nabla^d C_acbd
    auto nabla_D = partial_field<nu, scheme>(D) -
        rename_field<up<lambda>, dn<nu>, dn<a>>(Gamma) * rename_field<dn<lambda>, dn<b>, dn<c>>(D) -
        rename_field<up<lambda>, dn<nu>, dn<b>>(Gamma) * rename_field<dn<a>, dn<lambda>, dn<c>>(D) -
        rename_field<up<lambda>, dn<nu>, dn<c>>(Gamma) * rename_field<dn<a>, dn<b>, dn<lambda>>(D);
    auto raised_once = rename_field<up<c>, up<kappa>>(inverse) * rename_field<dn<kappa>, dn<lambda>>(Ricci);
    auto Ricci_up = raised_once * rename_field<up<lambda>, up<d>>(inverse);
    auto B = rename_field<up<c>, up<nu>>(inverse) * nabla_D + scale_field<1, 2>(Ricci_up * T);
    return BachExpressions<decltype(B), decltype(T)>{B, T};
}
