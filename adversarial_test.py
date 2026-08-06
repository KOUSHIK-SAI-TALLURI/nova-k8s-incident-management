from nova_rca import retrieve_similar_incidents, print_retrieved_incidents, generate_rca_analysis

incident = (
    "Pod memory usage has been climbing steadily over the past 6 hours, now at "
    "85% of a 2Gi memory limit — a limit that is already generous compared to "
    "what this service normally needs. No OOMKilled event or restart has "
    "occurred yet, but the trend suggests the limit will be hit within 24 hours "
    "if unaddressed. This same gradual climb pattern has been observed after "
    "each of the last three redeploys."
)

similar = retrieve_similar_incidents(incident)
print_retrieved_incidents(similar)

print("\n" + "="*60)
print("RCA OUTPUT")
print("="*60)
analysis = generate_rca_analysis(incident, similar)
print(analysis)
