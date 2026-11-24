**Title**: Bound active resources in a Context hierarchy

Add optional `max_resources` to Click Contexts. A nonnegative integer limits
resources registered through `with_resource()`; the default is unlimited.
Reject an additional registration with `RuntimeError` before entering its
context manager when a relevant limit is exhausted. Zero rejects the first
registration. Resource return values and normal cleanup behavior remain intact.

Limits apply to active, successfully entered resources. A Context's limit covers
its own registrations and all its descendants. A child inherits the parent's
numeric limit when omitted; an explicit child limit constrains that child's
subtree, without bypassing any ancestor's limit. Sibling registrations share
their ancestor's allowance, but do not consume a sibling's own subtree limit.

In-progress entries reserve capacity so reentrant resource acquisition during
entry cannot overrun the limit. Failed entry returns that reservation and does
not consume capacity. Closing a Context releases only its own
registered resources and returns their capacity to all affected limits, even if
cleanup raises. It must not reset capacity still held by another Context.
Repeated close does not release capacity twice, and closed Contexts can register
resources again. This feature does not make parent close recursively close its
children. Below the configured limit, existing command and completion behavior
must continue to work.
